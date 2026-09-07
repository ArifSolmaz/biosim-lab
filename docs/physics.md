# Fizik referansı / Physics reference

Bu belge platformdaki her formülü, kaynağını ve **nerede geçerli olmadığını**
listeler. Kod içindeki docstring'ler aynı DOI'leri taşır.

---

## 1. Akustik radyasyon kuvveti / Acoustic radiation force

### 1.1 Gor'kov potansiyeli

Küçük (`a ≪ λ`), sıkıştırılabilir, küresel bir parçacık için zaman-ortalamalı
potansiyel:

```
U = V_c · [ (f₁/2)·κ_f·⟨p²⟩  −  (3/4)·f₂·ρ_f·⟨v²⟩ ]

f₁ = 1 − κ_p/κ_f                        (monopol saçılma)
f₂ = 2(ρ_p − ρ_f)/(2ρ_p + ρ_f)          (dipol saçılma)
F  = −∇U
```

`exp(−iωt)` zaman çarpanı ve karmaşık genlikler için `⟨p²⟩ = |p|²/2` ve
`⟨v²⟩ = |∇p|²/(2(ρ_f ω)²)`.

**Kaynak:** Bruus (2012), *Acoustofluidics 7*, Lab Chip 12:1014,
`doi:10.1039/c2lc21068a`, denklem 17–19. Özgün türetme Gor'kov (1962), Sov.
Phys. Dokl. 6:773 (DOI öncesi).

**Uygulama:** `physics/acoustics.py::gorkov_potential`,
`::gorkov_force_on_grid`.

### 1.2 Kontrast faktörü ve 3 kat tuzağı

```
Φ_klasik = (5ρ_p − 2ρ_f)/(2ρ_p + ρ_f) − κ_p/κ_f          # contrast_factor()
Φ_Bruus  = f₁/3 + f₂/2 = Φ_klasik / 3                    # bruus_phi()
```

İki tanım da literatürde yaygındır ve **tam olarak 3 kat** farklıdır. Kuvvet
ifadesi hangi tanımı kullandığınıza göre değişir; karıştırmak bu alandaki en
yaygın sayısal hatadır. `tests/test_acoustics.py::test_classical_and_bruus_contrast_factors_differ_by_exactly_three`
oranı sabitler.

`Φ > 0` → basınç düğümüne; `Φ < 0` → karın noktasına. Sulu ortamdaki tüm
memeli hücreleri pozitif, lipit damlacıkları negatiftir; test her ikisini de
denetler.

### 1.3 Tek boyutlu duran dalga

```
F_r(x) = −(π p₀² V_c κ_f / 2λ) · Φ · sin(2k(x − x_düğüm)),   k = 2π/λ
```

`Φ` klasik tanımdadır. Bu ifade `F = 4π Φ_Bruus a³ k E_ac sin(2kx)` ile
özdeştir (`E_ac = p₀²κ_f/4`); türetme `physics/acoustics.py` docstring'inde.

**Uygulama:** `::primary_radiation_force_1d`.

### 1.4 SSAW düzeltmesi — dipol terimi (k_x/k_f)²

Ders kitabı ifadesi `k_x = k_f = ω/c_f` varsayar. Duran yüzey akustik dalgası
cihazında bu geçerli değildir: yanal dalga sayısını **taban** dayatır
(`k_x = 2π/λ_SAW`), sıvı ise Helmholtz denklemine uyar, dolayısıyla dikey
bileşen `k_y = √(k_f² − k_x²)` vardır — tam olarak Rayleigh açısındaki kırılım.

`p = p₀ sin(k_x x) cos(k_y y)` alanını Gor'kov potansiyeline koyunca,
`cos²(k_y y) = 1` düzleminde:

```
F_x = −(V p₀² κ_f k_x / 4) · sin(2k_x(x−x₀)) · [ f₁ + (3/2) f₂ (k_x/k_f)² ]
```

**Monopol terimi değişmez, dipol terimi `(k_x/k_f)²` ile ölçeklenir.**
`k_x = k_f` limitinde klasik `Φ = f₁ + 1.5f₂` birebir geri gelir.

MCF-7, su, 6.632 MHz, λ_SAW = 600 µm: Φ = 0.2371 → Φ_etkin = 0.1787.

**Uygulama:** `::effective_contrast_factor`.
**Doğrulama:** tam bir Helmholtz çözümüne karşı 7×10⁻⁶ bağıl RMS
(`test_gorkov_grid_force_reproduces_the_analytic_law_on_an_exact_field`).

### 1.5 Geçerlilik sınırı

Gor'kov, `k·a` cinsinden bir açılımın ilk terimidir; pratik sınır `k·a < 0.1`
(sıvıdaki dalga boyu ile). 6.632 MHz'de λ_su = 226 µm, 12 µm'lik bir hücre için
`k·a = 0.33` → **uyarı verilir**. 20 MHz'de `k·a ≈ 1` — Gor'kov ciddi biçimde
zorlanır. Bu, literatürdeki SSAW CTC cihazlarının da kabul ettiği bir
yaklaşıklıktır; platform sessiz kalmak yerine `RegimeWarning` üretir.

**Uygulama:** `::check_gorkov_validity`.

### 1.6 SSAW düğüm aralığı

Düğümler `λ_SAW/2` aralıklıdır, `λ_su/2` değil: sıvıdaki alan tabanın
periyodikliğini devralır.

**Kaynak:** Shi et al. (2009), Lab Chip 9:3354, `doi:10.1039/b910595f`;
Ding et al. (2012), PNAS 109:11105, `doi:10.1073/pnas.1209288109`.

128° YX LiNbO₃, `c_SAW = 3979 m/s` (`doi:10.1109/T-SU.1973.29761`):

| f | λ_SAW | düğüm aralığı | 300 µm kanaldaki düğüm |
|---|---|---|---|
| 6.632 MHz | 600 µm | 300 µm | 1 |
| 13.26 MHz | 300 µm | 150 µm | 2 |
| 20 MHz | 199 µm | 99.5 µm | 3 |

### 1.7 İkincil (Bjerknes) kuvvet — varsayılan kapalı

İki saçıcının birbirine yeniden saçması. `O(N²)` maliyetli ve bu projede
deneye karşı doğrulanmamıştır; `a ≪ d ≪ λ` varsayar.

**Kaynak:** Crum (1975), JASA 57:1363, `doi:10.1121/1.380614`.
**Uygulama:** `physics/secondary.py::secondary_bjerknes_force`.

---

## 2. Akışkanlar / Fluid mechanics

### 2.1 Dikdörtgen kanalda Poiseuille akışı

`|x| ≤ a`, `|y| ≤ b` kesitinde, eksenel basınç gradyanı `G = −dp/dz`:

```
u_z(x,y) = (16 b² G)/(μ π³) · Σ_{n tek} (−1)^((n−1)/2)/n³
           · [1 − cosh(nπx/2b)/cosh(nπa/2b)] · cos(nπy/2b)

Q = (64 b³ G)/(μ π⁴) · Σ_{n tek} [2a − (4b/nπ)·tanh(nπa/2b)]/n⁴
```

**Kaynak:** White, *Viscous Fluid Flow*, 3. baskı, §3-3.3; Bruus,
*Theoretical Microfluidics*, `doi:10.1093/oso/9780199235094.001.0001`, bölüm 3.

**Sayısal not:** yüksek en-boy oranlarında `cosh` taşar; oran log uzayında
hesaplanır (`cosh u/cosh v = e^(u−v)(1+e^(−2u))/(1+e^(−2v))`). 300 µm'den
100 mm'ye kadar test edilmiştir.

**Doğrulama:** profili sayısal integre edince `Q` bağıl hatası < 1e-3; geniş
kanal limitinde `u_max/u_ort → 1.5000`.

### 2.2 Stokes sürüklenme

```
F_d = 6πμr(u_f − u_p),   Re_p = ρ_f|u_rel|·2r/μ ≪ 1
```

**Kaynak:** Batchelor, `doi:10.1017/CBO9780511800955`, §4.9.

Varsayılan çalışma noktasında `Re_kanal = 0.53`, `Re_p = 0.15` — ikisi de
1'in altında. `Re > 1` olursa `RegimeWarning`.

### 2.3 Aşırı sönümlü limit

Parçacık hız gevşeme süresi `τ_p = 2ρ_p r²/(9μ)`. 18 µm'lik bir hücre için
~2×10⁻⁵ s; kanal geçiş süresi ~0.4 s. `Stk = τ_p/t_geçiş ≈ 5×10⁻⁵ ≪ 1`, bu
yüzden atalet düşürülür ve `v = u_f + F/(6πμr)` kullanılır. `mode="inertial"`
tam ODE'yi çözer; test ikisinin küçük `Stk`'de 1e-3 içinde uyuştuğunu gösterir.

### 2.4 Akustik akış (streaming) — modellenmiyor

Sınır tabakasındaki viskoz soğurmanın sürdüğü ikinci mertebe kararlı akış.
Büyüklüğü `~(3/16)u₁²/c`, parçacık boyutundan **bağımsızdır**; radyasyon
kuvveti ise `r³` ile ölçeklenir. Bir kritik yarıçapın altında streaming baskın
olur (suda 20 MHz'de ~1 µm), yani trombositler ve bakteriler için önemlidir,
9 µm'lik tümör hücreleri için değil.

**Kaynak:** Muller et al. (2012), Lab Chip 12:4617, `doi:10.1039/c2lc40612h`;
Rayleigh (1884), `doi:10.1098/rstl.1884.0002`.

Analitik yaklaşım `solvers/solver_openfoam::rayleigh_streaming_velocity`
içinde; tam çözüm Aşama 4'tür ve arka uç kurulu değilken platform şunu bildirir:
*"streaming devre dışı, analitik Rayleigh yaklaşımı kullanılıyor"*.

---

## 3. Elektro-kuasistatik / Electro-quasistatics

### 3.1 Karmaşık iletkenlik denklemi

```
∇·(σ* ∇φ) = 0,    σ* = σ + iω ε₀ ε_r
```

100 kHz–10 MHz'de 100 µm'lik bir elektrot açıklığı derinlemesine kuasistatiktir.

**Doğrulama:** paralel plaka geometrisinde `Z = L/(σ*A)` ile makine hassasiyeti
düzeyinde (bağıl fark < 1e-9) uyuşur.

### 3.2 Giaever–Keese hücre örtüsü modeli

```
1/Z_c = (1/Z_n)·[ Z_n/(Z_n+Z_m) + (Z_m/(Z_n+Z_m)) / (γI₀(γ)/(2I₁(γ)) + R_b(1/Z_n+1/Z_m)) ]
γ = α√(1/Z_n + 1/Z_m),    α = r_c √(ρ/h)
```

`R_b` hücreler arası (parasellüler) direnç [Ω·cm²], `α` ventral boşluk
parametresi [Ω^0.5·cm], `Z_m = 2/(iωC_m)` iki membran seri.

**Kaynak:** Giaever & Keese (1991), PNAS 88:7896, `doi:10.1073/pnas.88.17.7896`.

Birim sistemi: özgül empedanslar Ω·cm², alanlar cm², kapasitanslar F/cm².

### 3.3 Cell Index

```
CI(t) = (|Z_hücre(t)| − |Z_arka plan|) / Z_referans
```

`Z_referans` cihaz firmware'ine ait bir sabittir; 15 Ω varsayılanı **ASSUMPTION**
etiketlidir ve yalnızca y eksenini ölçekler.

### 3.4 Elektrot arayüzü

Sabit faz elemanı `Z = 1/(Q(iω)^n)`, `n ≈ 0.85–0.95` (yüzey pürüzlülüğü).
**Kaynak:** Franks et al. (2005), IEEE TBME 52:1295,
`doi:10.1109/TBME.2005.847523`. Yayılma direnci `R = 1/(4σa)`,
`a = √(A/π)` — Newman (1966), `doi:10.1149/1.2424009`.

---

## 3b. Sıcaklık ve hücre hasarı / Temperature and cell damage

### 3b.1 Sıcaklığa bağlı su özellikleri

Malzeme kütüphanesindeki değerler tek bir sıcaklıkta (sulu ortamlar için 25 °C)
verilmiştir. Bu bir çizelge için yeterli, bir simülasyon için yanlıştır:

| Özellik | 25 °C | 37 °C | değişim |
|---|---|---|---|
| viskozite `μ` | 0.890 mPa·s | 0.691 mPa·s | **−22 %** |
| ses hızı `c` | 1496.7 m/s | 1523.6 m/s | +1.8 % |
| yoğunluk `ρ` | 997.0 kg/m³ | 993.3 kg/m³ | −0.4 % |

Akustoforetik hız `F/(6πμr)` olduğundan, tek başına viskozite düşüşü hücreleri
tezgâhtakinden **%25 daha hızlı** hareket ettirir (sıkıştırılabilirlik değişimi
kısmen dengeler). Tezgâhta ayarlanıp inkübatörde çalıştırılan bir cihaz aynı
cihaz değildir.

**Korelasyonlar** (üçü de 25 °C'de kütüphane değerlerini dört anlamlı basamağa
kadar geri verir — `tests/test_environment.py` bunu denetler):

* yoğunluk: Kell (1975), `doi:10.1021/je60064a005`
* ses hızı: Marczak (1997), `doi:10.1121/1.420332`
* viskozite: Swindells/CRC bağıntısı, Kestin ve ark. (1978) ile uyumlu,
  `doi:10.1063/1.555581`

Suyun 4 °C'deki yoğunluk anomalisi testle doğrulanır — polinomun doğru
aktarıldığının güçlü bir göstergesidir.

**Uygulama:** `core/environment.py::fluid_at`.

### 3b.2 Soğurma ve ısınma

Suda basınç soğurma katsayısı `α = 2.2×10⁻³ (f/MHz)² dB/cm`
(Pinkerton 1949, `doi:10.1088/0370-1301/62/2/307`):

| f | α | 300 µm'de kayıp |
|---|---|---|
| 6.632 MHz | 1.1 Np/m | %0.03 |
| 20 MHz | 10.1 Np/m | %0.3 |

Yani **sıvı pratik olarak kayıpsızdır**; Helmholtz çözücüsündeki sönüm
parametresi suyun değil *cihazın* kalite faktörünün karşılığıdır.

Duran dalgada birim hacimde ısı: `q = α p₀²/(2ρc)`. Varsayılan çalışma
noktasında bu 72 kW/m³ olur ama kanal hacmi 3×10⁻¹¹ m³ olduğundan toplam güç
2.2 µW'tır ve akan sıvı bunu **0.006 K** ısıtır — ihmal edilebilir, ve bu artık
*varsayılmıyor, hesaplanıyor*.

Gerçek cihazlardaki ısınma sudan değil, IDT parmaklarındaki dirençsel ve
tabandaki viskoelastik kayıplardan gelir. Bu proje onu hesaplayamaz (Aşama 4
piezoelektrik çözümü gerekir), bu yüzden `TRANSDUCER_HEATING_K_PER_W = 12 K/W`
açıkça **ASSUMPTION** etiketlidir. Kendi çipinizi termoçiftle ölçüp
`temperature` değerini doğrudan verin.

### 3b.3 Hücre hasarı — üç mekanizma

Akustik ayrımın satış argümanı nazik olmasıdır. "Nazik" bir iddiadır ve iddia
hesaplanabilir bir şeydir.

**Isı — CEM43 termal dozu.** Hasar sıcaklıkla doğrusal değil birikir; standart
para birimi 43 °C'deki eşdeğer dakikadır:

```
CEM43 = Σ R^(43−T) · Δt      R = 0.25 (T < 43 °C), 0.5 (T ≥ 43 °C)
```

Sapareto & Dewey (1984), `doi:10.1016/0360-3016(84)90379-1`; eşik derlemesi
van Rhoon ve ark. (2013), `doi:10.1007/s00330-013-2825-y`.

**Kayma gerilmesi.** Akış zarı çeker. Eritrositler ~150 Pa sürekli kayma
üzerinde hemoliz olur — Leverett ve ark. (1972),
`doi:10.1016/S0006-3495(72)86085-5`. Varsayılan kanalda tepe duvar kayması
**0.66 Pa**'dır; üstelik odaklanmış hücre kanalın en nazik yeri olan merkezde
oturur.

**Kavitasyon.** Mekanik indeks `MI = p_negatif[MPa]/√(f[MHz])`; tanısal
ultrasonda üst sınır 1.9 — Apfel & Holland (1991),
`doi:10.1016/0301-5629(91)90125-Q`. Varsayılan noktada **MI = 0.175**.

**Varsayılan çalışma noktasındaki marjlar:**

| Mekanizma | Gösterge | Eşik | Marj |
|---|---|---|---|
| Isı | 3.5×10⁻¹³ CEM43 dk | 15 CEM43 dk | > 10⁴× |
| Kayma | 0.59 Pa | 150 Pa | 254× |
| Kavitasyon | MI 0.175 | MI 1.9 | 10.9× |

Baskın neden **maruziyet süresidir**: hücreler alanda 0.36 saniye kalır.
Aynı şiddette dakikalarca tutan bir tuzak bambaşka bir meseledir — test paketi
bunu açıkça karşılaştırır.

**Modellenmeyen:** liziz eşiğinin altındaki membran gözeneklenmesi
(sonoporasyon) — tripan mavisini öldürmeden içeri alabilir ve canlılık okumasını
bozar; ve ölü bir hücrenin akustik özelliklerinin değişmesi. Ölü hücreler canlı
hücre yoğunluğu ve sıkıştırılabilirliğiyle taşınır (açıkça etiketli varsayım),
bu yüzden ölü hücrelerin nereye gittiği tahmini canlılarınkinden daha az
güvenilirdir.

**Uygulama:** `instruments/saw_sorter/viability.py`.

---

## 4. Görüntü analizi / Image analysis

### 4.1 Watershed segmentasyon

Gauss arka plan çıkarma → Otsu eşikleme → alan kapısı → mesafe dönüşümü
tepeleriyle tohumlanmış watershed.

**Kaynak:** Beucher & Meyer (1993), *The morphological approach to
segmentation: the watershed transformation*.

### 4.2 Neubauer geometrisi

Oda derinliği 0.100 mm, köşe karesi 1 mm × 1 mm → 1×10⁻⁴ mL. Buradan klasik
"10⁴ ile çarp" kuralı gelir.

```
konsantrasyon [hücre/mL] = N / (A_görüş · derinlik) · seyreltme
```

Poisson sayım belirsizliği `1/√N`: 100 hücre saymak %10 standart hata demektir —
çoğu zaman ölçmeye çalıştığınız farktan büyüktür.

### 4.3 Takip ve hareketlilik

Crocker–Grier en yakın komşu bağlama (`trackpy`),
`doi:10.1006/jcis.1996.0217`.

```
MSD(τ) = ⟨|r(t+τ) − r(t)|²⟩ = 4Dτ^α
```

`α = 1` difüzyon, `α → 2` balistik/yönlü, `α < 1` alt-difüzyon.
**Kaynak:** Selmeczi et al. (2005), Biophys. J. 89:912,
`doi:10.1529/biophysj.105.061150`.

Sentetik demo `α = 1.89` verir (persistans 0.85 ile üretilmiş kalıcı rastgele
yürüyüş) — beklendiği gibi süperdifüzif.

---

## 5. Bilinen modelleme sınırları

1. **Piezoelektrik tam çözüm yok.** Taban yer değiştirmesi `u₀` bir *girdidir*;
   voltaj→basınç dönüşümü `PRESSURE_PER_VOLT_ASSUMPTION` ile doğrusal kabul
   edilir. Kaldırmak Aşama 4'ün (Elmer) işidir.
2. **Streaming ihmal edilir** (yukarıda §2.4).
3. **1-B analitik model yükseklik zayıflamasını yok sayar.** Taban düzlemi
   genliğini her `y` için uygular; 50 µm kanalda gerçek yanal kuvvet kanal
   ortalamasında bu değerin %53'üdür. Analitik mod bu yüzden **iyimserdir**;
   FEM modu gerçekçi olandır.
4. **Dikey Gor'kov kuvveti varsayılan kapalı.** 2-B kesit modelinde dengeleyici
   bir kaldırma kuvveti yoktur, bu yüzden açılırsa hücreler eksenel hızın sıfır
   olduğu duvara yığılır ve çıkıştan geçmezler.
5. **Hücreler küresel sıkıştırılabilir damlacık** olarak modellenir; membran
   ve çekirdek ayrı ayrı temsil edilmez.
6. **Parçacık–parçacık etkileşimi kapalıdır** (§1.7).
7. **RTCA tek katmanlı kabuk modeli** kullanır; `R_b` tek serbest parametredir.
