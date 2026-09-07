# biosim-lab — Architecture / Mimari

> Açık kaynak, eklenti mimarili "sanal laboratuvar cihazı" platformu.
> Open-source, plugin-based *virtual laboratory instrument* platform.

---

## 1. Tasarım ilkeleri / Design principles

| # | İlke | Karşılığı kodda |
|---|------|-----------------|
| 1 | **Çekirdek hafif kalır.** Ağır çözücüler (OpenFOAM, Elmer, FEniCSx, Napari) asla zorunlu bağımlılık değildir. | `core/plugin.py::optional_import`, `pyproject.toml` extras |
| 2 | **Her cihaz bir eklentidir.** Tek arayüz: `Instrument`. | `core/plugin.py::Instrument` |
| 3 | **Her çözücü bir eklentidir.** Tek arayüz: `Solver`. | `core/solver.py::Solver` |
| 4 | **Birimler açıktır.** Tüm genel API `pint` ile denetlenir; iç hesap SI'de saf `float`/`ndarray`. | `core/units.py` |
| 5 | **Veri tek biçimlidir.** Alanlar `xarray.Dataset`, tablolar `pandas.DataFrame`, disk `NetCDF`/`Parquet`. | `core/io.py` |
| 6 | **Her formülün kaynağı vardır.** Fizik fonksiyonlarının docstring'inde DOI. Kaynağı olmayan sayı `ASSUMPTION` etiketli. | `core/materials.py`, `instruments/*/physics/` |
| 7 | **Keşif çalışma zamanındadır.** `entry_points` üzerinden; kurulu olmayan eklenti hata değil, eksik yetenektir. | `core/plugin.py::discover_instruments` |

---

## 2. Katmanlar / Layers

```
        ┌──────────────────────────────────────────────────────┐
CLI /   │  cli.py      biosim run|dashboard|doctor|list|sweep   │
Panel   │  core/viz/dashboard.py  (Panel + Plotly, tarayıcı)    │
        └───────────────────────┬──────────────────────────────┘
                                │ Instrument ABC
        ┌───────────────────────▼──────────────────────────────┐
INSTRU- │ saw_sorter │ impedance_rtca │ cell_counter │ tracker  │
MENTS   │  (Aşama 1) │   (Aşama 2)    │  (Aşama 3)   │ (Aşama 3)│
        └───────────────────────┬──────────────────────────────┘
                                │ core services
        ┌───────────────────────▼──────────────────────────────┐
CORE    │ config │ geometry │ fem │ solver │ particles │        │
        │ materials │ io │ viz │ units │ plugin │ registry     │
        └───────────────────────┬──────────────────────────────┘
                                │ Solver ABC (optional back-ends)
        ┌───────────────────────▼──────────────────────────────┐
SOLVERS │ builtin (scikit-fem, analytic) │ openfoam │ elmer     │
        │        ZORUNLU                 │   opsiyonel (Aşama 4)│
        └──────────────────────────────────────────────────────┘
```

Ok yönü tek yönlüdür: **instruments → core → solvers**. Çekirdek hiçbir zaman bir
cihaz eklentisini `import` etmez; cihazlar birbirini `import` etmez.

---

## 3. Dizin haritası / Directory map

```
biosim_lab/
  core/
    units.py       # pint UnitRegistry (tek örnek), Q_ kısayolu, SI dönüşüm yardımcıları
    environment.py # sıcaklığa bağlı su özellikleri, soğurma, ısı bütçesi
    config.py      # pydantic v2 deney tanımı; YAML ↔ nesne; ExperimentConfig
    materials.py   # Fluid / CellType / Substrate kütüphanesi + DOI + ASSUMPTION etiketleri
    geometry.py    # Gmsh sarmalayıcı: straight_channel, idt_electrodes, well_plate
    fem/
      helmholtz.py       # ∇²p + k²p = 0  (akustik basınç)
      electroquasistatic.py  # ∇·(σ + iωε)∇φ = 0  (empedans)
      stokes.py          # Stokes akışı (düşük Re)
    solver.py      # Solver(ABC): setup/run/fields; SolverResult; registry + doctor
    particles.py   # ForceRegistry, ParticleState, LagrangianTracker (solve_ivp, vektörize)
    io.py          # save/load NetCDF+Parquet; RTCA CSV/xlsx, TIFF yığını okuyucuları
    viz/
      fields3d.py   # PyVista
      curves.py     # Plotly (eğri, heatmap, histogram, Nyquist/Bode, plaka haritası)
      dashboard.py  # Panel panosu (kaydırıcılar → canlı yeniden hesap)
      napari_layers.py  # opsiyonel Napari katmanları
    plugin.py      # Instrument(ABC), optional_import, discover_instruments
  instruments/
    saw_sorter/    # AŞAMA 1 — tam
      viability.py            # CEM43 termal doz, kayma, kavitasyon; canlı/ölü
      physics/acoustics.py    # Gor'kov, ARF, kontrast faktörü, SAW alanı
      physics/drag.py         # Stokes sürüklenme, Re denetimi
      physics/secondary.py    # Bjerknes, yerçekimi/kaldırma, duvar itme (varsayılan kapalı)
      flow.py                 # Dikdörtgen kanal Poiseuille (Fourier serisi)
      fem_model.py            # Gmsh + scikit-fem Helmholtz; sızıntılı SAW sınır koşulu
      simulate.py             # Lagrangian koşu, metrikler, parametre taraması
      instrument.py           # SAWSorter(Instrument)
    impedance_rtca/  # AŞAMA 2 — iskelet + çalışan minimal örnek
    cell_counter/    # AŞAMA 3 — iskelet + sentetik demo
    cell_tracker/    # AŞAMA 3 — iskelet + sentetik demo
  solvers/
    solver_openfoam/ # AŞAMA 4 — yalnızca iskelet + C++/controlDict şablonları
    solver_elmer/    # AŞAMA 4 — yalnızca iskelet + .sif şablonu
  cli.py
```

---

## 4. `Instrument` sözleşmesi

```python
class Instrument(ABC):
    name: str                  # entry_point adı, örn. "saw_sorter"
    display_name: str
    description: str
    ConfigModel: type[BaseModel]   # cihaza özgü pydantic bloğu

    def __init__(self, config: ExperimentConfig) -> None
    def setup(self) -> None                  # mesh/alan/materyal hazırlığı, idempotent
    def run(self) -> InstrumentResult        # ağır hesap
    def results(self) -> InstrumentResult    # son sonuç (run çağrılmadıysa hata)
    def dashboard(self) -> "panel.viewable.Viewable"   # etkileşimli pano
    @classmethod
    def example_config(cls) -> dict          # `biosim init <name>` için
```

`InstrumentResult` = `xarray.Dataset` (alanlar + yörüngeler) + `dict` (skaler metrikler)
+ `pandas.DataFrame` (parçacık tablosu). Tek bir `.nc` dosyasına yazılabilir; metrikler
NetCDF `attrs` içinde JSON olarak taşınır.

Yaşam döngüsü: `setup() → run() → results() → dashboard()`.
`setup()` iki kez çağrılırsa yeniden hesaplamaz (idempotent).

---

## 5. `Solver` sözleşmesi

```python
class Solver(ABC):
    name: str
    kind: Literal["acoustics","electro","flow","piezo","streaming"]
    def is_available(self) -> tuple[bool, str]      # (kurulu mu, açıklama)
    def setup(self, mesh, bc: dict) -> None
    def run(self) -> None
    def fields(self) -> xr.Dataset                  # SI birimli alanlar
```

Kurulu değilse `is_available() → (False, sebep)`; çağıran taraf yedek analitik yola düşer
ve **uyarır**: `"streaming devre dışı, analitik Rayleigh yaklaşımı kullanılıyor"`.

---

## 6. Eklenti keşfi / Plugin discovery

`pyproject.toml`:

```toml
[project.entry-points."biosim_lab.instruments"]
saw_sorter      = "biosim_lab.instruments.saw_sorter:SAWSorter"
impedance_rtca  = "biosim_lab.instruments.impedance_rtca:ImpedanceRTCA"
cell_counter    = "biosim_lab.instruments.cell_counter:CellCounter"
cell_tracker    = "biosim_lab.instruments.cell_tracker:CellTracker"

[project.entry-points."biosim_lab.solvers"]
builtin_helmholtz = "biosim_lab.core.fem.helmholtz:HelmholtzSolver"
openfoam          = "biosim_lab.solvers.solver_openfoam:OpenFOAMSolver"
elmer             = "biosim_lab.solvers.solver_elmer:ElmerSolver"
```

Üçüncü taraf bir paket aynı grup adını kullanarak kendi cihazını ekleyebilir; çekirdekte
değişiklik gerekmez. `biosim doctor` kurulu tüm eklentileri ve arka uçları raporlar.

---

## 7. Veri akışı / Data flow (saw_sorter örneği)

```
config.yaml
   │  pydantic doğrulama + birim denetimi
   ▼
ExperimentConfig ──► materials.py (MCF-7, RBC: ρ, β, r-dağılımı, DOI)
   │
   ├─ mode="analytic" ─► acoustics.standing_wave_pressure(x)  ─┐
   │                                                           │
   └─ mode="fem" ─► geometry.straight_channel (Gmsh)           │
                     └► fem/helmholtz.py (scikit-fem)          │
                        └► p(x,y) ─► ∇ ile Gor'kov kuvveti ────┤
                                                               ▼
                                          ForceRegistry: ARF + drag + (opsiyonel)
                                                               │
                                              particles.py  solve_ivp (vektörize)
                                                               ▼
                                     trajectories → metrics (verim, saflık, histogram)
                                                               ▼
                                     io.py → results.nc / .parquet / .csv
                                                               ▼
                                     viz/ → PyVista alan, Plotly heatmap, Panel pano
```

---

## 8. Birim politikası / Unit policy

- **Sınırda birim, içeride SI.** `config.py` `"20 MHz"` gibi dizeleri `pint` ile ayrıştırır,
  `to_base_units()` uygular, çekirdeğe saf `float` (Hz) geçirir.
- Fizik fonksiyonları **birimsiz `float`/`ndarray` alır ve SI döndürür**; docstring'de
  her argümanın birimi yazar. Bu, `solve_ivp` içinde `pint` yükünden kaçınır.
- `core/units.py` tek bir `UnitRegistry` örneği yayar (`ureg`, `Q_`). İki registry
  karışımı `pint`'te hataya yol açtığı için başka yerde `UnitRegistry()` çağrılmaz.

---

## 9. Doğrulama stratejisi / Validation strategy

| Katman | Test |
|--------|------|
| Kuvvet formülleri | Analitik özel durumlar; kontrast faktörü işareti (MCF-7 > 0 → düğüme, lipit < 0 → antinoda) |
| Rejim | `Re = ρ u d/μ << 1` denetimi; `Stk << 1`; ihlalde `RegimeWarning` |
| Analitik ↔ FEM | `examples/04_validate_analytic_vs_fem.py`: aynı parametrelerle kuvvet profili RMS farkı; eşik %30, gözlenen %12.8. Ayrım metriklerinde iki mod yalnızca **sıralamada** uyuşur, sayılarda değil — bkz. sınırlama 3 |
| Gor'kov ↔ kapalı form | Tam bir Helmholtz alanı üzerinde ızgara kuvveti vs 1-B yasa: RMS 7×10⁻⁶ |
| Elektro-kuasistatik | Paralel plaka `Z = L/(σ*A)`: bağıl fark < 1e-9 |
| Yakınsama | FEM kuvvet alanının antisimetri artığı çözünürlükle 0.118 → 0.009 |
| Kütle korunumu | Parçacık sayısı korunur; kanal dışına kaçan parçacık yok |
| Eklenti izolasyonu | OpenFOAM/Elmer kurulu değilken tüm paket geçer |
| Streaming (opsiyonel) | Kurulu ise 2D Rayleigh streaming analitik çözümü ile karşılaştırma |

---

## 10. Aşama planı / Roadmap

| Aşama | Kapsam | Durum |
|-------|--------|-------|
| 1 | `saw_sorter` tam: analitik + FEM, metrikler, tarama, pano, testler | **Tam** |
| 2 | `impedance_rtca`: Giaever–Keese kabuk modeli, Cell Index, IC50, RTCA parser | **Minimal çalışır** |
| 3 | `cell_counter`, `cell_tracker`: watershed + trackpy, sentetik veri üreteci | **İskelet + demo** |
| 4 | `solver_openfoam`, `solver_elmer` | **Yalnızca iskelet + şablon** |

---

## 11. Bilinen sınırlamalar / Known limitations

Bunların hepsi kodda da belgelidir; buradaki liste tam kümedir.

1. **Piezoelektrik tam çözüm yoktur.** LiNbO₃ yüzey yer değiştirmesi `u₀` bir *girdidir*;
   voltaj→basınç dönüşümü `PRESSURE_PER_VOLT_ASSUMPTION` ile doğrusal kabul edilir
   (15 Vpp → 0.45 MPa). Bu, sistemdeki tek en büyük kaynaksız girdidir. Kaldırmak Aşama 4
   (Elmer) işidir; o zamana kadar kendi çipiniz için `pressure_amplitude`'ı ölçüp verin.

2. **Akustik streaming ihmal edilir.** Streaming hızı `~(3/16)u₁²/c` parçacık boyutundan
   bağımsızdır, radyasyon kuvveti ise `r³` ile ölçeklenir; bir kritik yarıçapın altında
   (suda 20 MHz'de ~1 µm) streaming baskın olur. 9 µm'lik tümör hücreleri için ikincil,
   trombosit ve bakteriler için değil. Bkz. Muller et al. (2012),
   doi:10.1039/c2lc40612h. Arka uç yokken platform *"streaming devre dışı, analitik
   Rayleigh yaklaşımı kullanılıyor"* uyarısını verir.

3. **1-B analitik model yükseklik zayıflamasını yok sayar — dolayısıyla iyimserdir.**
   Kapalı form kuvveti her `y` için taban düzlemi genliğiyle uygular. Gerçek sızıntılı SAW
   alanı yükseklikle zayıflar: 50 µm kanalda yanal kuvvet tavanda taban değerinin ~%19'una
   düşer, kanal ortalaması **%53**'tür. `mode="fem"` gerçekçi olandır ve daha düşük geri
   kazanım verir; fark `examples/04_validate_analytic_vs_fem.py` ile sayısallaştırılır.

4. **FEM'in dikey Gor'kov bileşeni varsayılan kapalıdır** (`enable_vertical_arf=False`).
   2-B kesit modelinde onu dengeleyecek bir kaldırma kuvveti (ataletsel lift, duvar
   yağlaması, çökelme) yoktur; açılırsa hücreler eksenel hızın sıfır olduğu duvara yığılır
   ve çıkıştan hiç geçmezler. Açarsanız `all_cells_exited` metriğini denetleyin.

5. **FEM yan duvar modeli kanal genişliğine bağlıdır.** Varsayılan `wall="ssaw"` yan
   duvarları rijit alır; bu yalnızca kanal genişliği `λ_SAW/2`'nin tam katıyken tutarlıdır
   (duvarlar basınç karın noktalarına düşer, `∂p/∂x = 0` zaten sağlanır). Aksi halde
   `check_wall_consistency()` uyarır ve `wall="pdms"` kullanılmalıdır.

6. **Hücreler küresel, sıkıştırılabilir sıvı damlacığı** olarak modellenir (Gor'kov
   varsayımı `k·a ≪ 1`). Varsayılan senaryoda `k·a = 0.39`, 20 MHz'de `k·a ≈ 1` —
   her ikisinde de uyarı verilir. Membran ve çekirdek ayrı temsil edilmez.

7. **Parçacık–parçacık etkileşimi (sekonder Bjerknes) varsayılan kapalıdır**; `O(N²)`
   maliyetlidir ve bu projede deneye karşı doğrulanmamıştır.

8. **`impedance_rtca` tek katmanlı kabuk modeli** kullanır; `R_b` tek serbest
   parametredir. Endpoint IC50 maruziyet süresine bağlıdır ve varsayılan senaryoda ekilen
   değerin altında okur — bu gerçek bir endpoint deneyinin davranışıdır.

9. **IDT ısınması hesaplanmaz.** Suyun sesi soğurmasından gelen ısı ilk
   ilkelerden hesaplanır ve varsayılan noktada 0.006 K çıkar — ihmal edilebilir,
   ve artık bunu *görebiliyorsunuz*. Gerçek cihazlardaki birkaç ila onlarca
   kelvinlik ısınma IDT'deki dirençsel ve tabandaki viskoelastik kayıplardan
   gelir; bunu hesaplamak Aşama 4'ün piezoelektrik çözümünü gerektirir.
   `TRANSDUCER_HEATING_K_PER_W` açıkça varsayım etiketlidir.

10. **Sonoporasyon ve ölü hücre akustiği modellenmez.** Liziz eşiğinin
    altındaki membran gözeneklenmesi tripan mavisi okumasını bozabilir; ölü
    hücreler canlı hücre yoğunluğu ve sıkıştırılabilirliğiyle taşınır, bu yüzden
    varış yerleri canlılarınkinden daha az güvenilirdir.

11. **Gmsh opsiyoneldir** (`pip install biosim-lab[mesh]`). Kurulu değilse düz kanal
   şablonu tam olarak eşdeğer bir yapılandırılmış üçgenlemeye düşer; yalnızca PDMS duvar
   katmanı Gmsh gerektirir. Bazı platformlarda (linux/arm64) Gmsh wheel'i yoktur, bu
   yüzden çekirdek bağımlılığı değildir.
