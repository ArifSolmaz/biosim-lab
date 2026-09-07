# Doğrulama / Validation

Neyin, neye karşı, hangi toleransla doğrulandığı. Hepsi `pytest` ile
çalıştırılır ve opsiyonel arka uçlar kurulu olmadan geçer.

## Kapalı form karşılaştırmaları / Closed-form checks

| Ne | Neye karşı | Sonuç | Test |
|---|---|---|---|
| Elektro-kuasistatik FEM | Paralel plaka `Z = L/(σ*A)` | bağıl fark **< 1e-9** | `test_fem.py::test_electroquasistatic_matches_the_parallel_plate_formula` |
| Gor'kov ızgara kuvveti | 1-B kapalı form (tam Helmholtz alanı üzerinde) | RMS **7×10⁻⁶** | `test_acoustics.py::test_gorkov_grid_force_reproduces_the_analytic_law_on_an_exact_field` |
| Helmholtz FEM | `cos(kx)` mod biçimi | RMS **< 2 %** | `test_fem.py::test_helmholtz_reproduces_a_one_dimensional_standing_wave` |
| Poiseuille serisi | Profilin sayısal integrali | `Q` bağıl hatası **< 1e-3** | `test_flow.py::test_flow_rate_is_conserved_by_the_series` |
| Poiseuille serisi | Paralel plaka limiti `u_max/u_ort = 3/2` | **1.5047** (W/H = 200) | `test_flow.py::test_wide_channel_recovers_the_parallel_plate_ratio` |
| Aşırı sönümlü izleyici | Terminal hız `F/(6πμr)` | bağıl fark **< 1e-4** | `test_particles.py::test_overdamped_particle_reaches_terminal_velocity` |
| Atalet vs aşırı sönümlü | Birbirine (küçük `Stk`) | fark **< 1e-3** | `test_particles.py::test_inertial_mode_agrees_with_overdamped_for_tiny_stokes_number` |
| 4PL IC50 uydurma | Gürültüsüz sentetik eğri | IC50 bağıl hata **< 1e-3** | `test_impedance_rtca.py::test_ic50_is_recovered_from_noiseless_data` |
| MSD üsteli | `MSD ∝ τ` ve `MSD ∝ τ²` | `α = 1.000` / `2.000` | `test_image_instruments.py` |

## Özdeşlikler ve işaretler / Identities and signs

* `Φ_klasik = 3·Φ_Bruus` — tam olarak, her malzeme için.
* Nötr parçacık için `Φ = 0`.
* `k_x = k_f` limitinde `effective_contrast_factor` → klasik `Φ`.
* Kuvvet düğümlerde ve karın noktalarında sıfır; düğümün solunda `+`, sağında `−`.
* Kuvvet hacimle doğrusal, `p₀²` ile, `1/λ` ile ölçeklenir.
* Tüm memeli hücreleri için `Φ > 0`, lipit damlacığı için `Φ < 0`.
* Rayleigh açısı Snell yasasını sağlar (`22.1°`), `c_f ≥ c_SAW` ise hata verir.

## Korunum / Conservation

* Parçacık sayısı korunur; hiçbir hücre kaybolmaz.
* Tüm yörüngeler kanal sınırları içindedir.
* Varsayılan senaryoda `all_cells_exited = True`; dikey kuvvet açılırsa bu
  bayrak `False` olur ve sorunu görünür kılar.

## Analitik ↔ FEM (Aşama 1)

`examples/04_validate_analytic_vs_fem.py`, aynı parametrelerle iki yolu
karşılaştırır ve farkı açıklar.

```
  cell       |F| analytic      |F| FEM    ratio      RMS
  mcf7           129.57 pN     106.70 pN    0.824   12.77 %
  rbc              5.38 pN       4.43 pN    0.824   12.76 %

  basınç profili RMS farkı: 0.00 % (p₀ cinsinden)

  yanal kuvvetin yükseklikle zayıflaması:
    y =  0 µm  106.70 pN       y = 30 µm   41.93 pN
    y = 10 µm   88.48 pN       y = 40 µm   25.64 pN
    y = 20 µm   64.86 pN       y = 50 µm   19.92 pN
    kanal ortalaması / taban düzlemi: 0.532
```

**Tolerans:** kuvvet profilinde %30 RMS. Gözlenen %12.8 → PASS.

**Fark neden var:**
1. 1-B model taban düzlemi genliğini her yükseklikte uygular; gerçek alan
   yükseklikle zayıflar (kanal ortalaması %53). Baskın neden budur ve ayrım
   metriklerinde kuvvet profilinden daha büyük fark yaratır.
2. FEM tavanı PDMS empedans sınırıdır, rijit duvar değil; sızıntılı dalganın bir
   kısmı kaçar ve `⟨v²⟩`'ye ilerleyen bir bileşen ekler.
3. FEM genliği `p₀`'a yeniden ölçeklenir, çünkü sürülen bir rezonatörün mutlak
   yanıtı uydurulan bir kalite faktörüne bağlıdır.

Test, iki modun **sıralamayı** aynı verdiğini ve FEM'in analitikten daha
iyimser **olamayacağını** zorunlu kılar — sayıların eşitliğini değil.

## Yakınsama / Convergence

FEM kuvvet alanının antisimetri artığı (fiziksel olarak sıfır olmalı):

| çözünürlük | tepe \|F_x\| | antisimetri artığı |
|---|---|---|
| 32 | 109.60 pN | 0.118 |
| 64 | 106.70 pN | 0.021 |
| 128 | 106.23 pN | 0.009 |

Yapılandırılmış üçgen ağın köşegen yönü sol–sağ simetrisini kırar; artık
çözünürlükle düzenli olarak azalır, yani ayrıklaştırma hatasıdır.

## Rejim denetimleri / Regime guards

Her koşuda hesaplanır ve ihlalde `RegimeWarning` verilir:

| Nicelik | Sınır | Varsayılan senaryodaki değer |
|---|---|---|
| `k·a` (Gor'kov) | < 0.1 | 0.39 → **uyarı** |
| `Re_kanal` | < 1 | 0.53 |
| `Re_parçacık` | < 1 | 0.15 |
| `Stk` | < 0.1 | ~5×10⁻⁵ |
| kanaldaki düğüm sayısı | 1 | 1 (6.632 MHz) / 3 (20 MHz → **uyarı**) |

## Eklenti izolasyonu / Plugin isolation

`tests/test_solver_plugins.py`, OpenFOAM ve Elmer **kurulu değilken**:

* modüllerin sorunsuz import edildiğini,
* `is_available()`'ın neyin eksik olduğunu söylediğini,
* `get_solver()`'ın `ImportError` değil `UnavailableSolver` döndürdüğünü,
* `setup()`'ın şablonların yerini gösteren `NotImplementedError` verdiğini,
* şablonların pakete dahil edildiğini

doğrular. Kuruluysa 2-B Rayleigh streaming analitik çözümüne karşı doğrulama
testi `optional_backend` işaretiyle çalışır (şu an köprü yazılmadığı için
`skip` eder ve o işin kabul kapısıdır).

## Sentetik yer gerçeği / Synthetic ground truth

Görüntü cihazları, doğru cevabın tam olarak bilindiği sentetik veriyle
puanlanır:

| Ölçüt | Sonuç |
|---|---|
| Segmentasyon geri çağırma | %84 (kenara değen ve birleşen hücreler hariç) |
| Takip geri kazanımı | 67 / 60 iz (parçalanma nedeniyle > 1) |
| Ölçülen hız | 0.1446 µm/dk (gerçek 0.1430 — **%1 hata**) |
| MSD üsteli | 1.89 (kalıcı yürüyüş, beklendiği gibi süperdifüzif) |
