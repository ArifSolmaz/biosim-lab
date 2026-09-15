# biosim-lab

**Açık kaynak, eklenti mimarili sanal laboratuvar cihazı platformu**
**Open-source, plugin-based virtual laboratory instrument platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Ticari cihazların (xCELLigence RTCA, akustik hücre ayırıcılar, otomatik hücre
sayıcılar, canlı hücre takip sistemleri) yaptığı analizleri **tamamen açık
kaynak** araçlarla simüle eder ve gerçek cihaz verisiyle de çalışır.

Simulates what commercial bio-instruments do — impedance-based real-time cell
analysis, acoustic cell sorting, automated counting, live-cell tracking — using
**only** free and open-source Python tooling, and runs the same analyses on
measured instrument data.

> Ticari yazılım kullanılmaz: COMSOL, ANSYS ve MATLAB yoktur. Yığın NumPy,
> SciPy, xarray, pint, pydantic, Gmsh, scikit-fem, PyVista, Plotly + Panel,
> scikit-image ve trackpy'dir. Ağır çözücüler (OpenFOAM, Elmer) yalnızca
> **opsiyonel** eklentidir; çekirdek onlarsız eksiksiz çalışır.

---

## İçindekiler / Contents

- [Kurulum / Installation](#kurulum--installation)
- [Hızlı başlangıç / Quick start](#hızlı-başlangıç--quick-start)
- [Cihazlar / Instruments](#cihazlar--instruments)
- [Çalışma kipleri / Operating modes](#çalışma-kipleri--operating-modes)
- [Literatür doğrulaması / Literature benchmarks](#literatür-doğrulaması--literature-benchmarks-aşama-1b)
- [Fizik / Physics](#fizik--physics)
- [Bilmeniz gereken beş sonuç / Five findings you should know](#bilmeniz-gereken-beş-sonuç--five-findings-you-should-know)
- [Örnekler / Examples](#örnekler--examples)
- [Komut satırı / CLI](#komut-satırı--cli)
- [Docker](#docker)
- [Kaynaklar ve varsayımlar / Provenance and assumptions](#kaynaklar-ve-varsayımlar--provenance-and-assumptions)
- [Katkı / Contributing](#katkı--contributing)

---

## Kurulum / Installation

Python ≥ 3.11 gerekir.

```bash
git clone https://github.com/biosim-lab/biosim-lab.git
cd biosim-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[mesh]"   # veya sadece: pip install -e .
biosim doctor              # neyin kurulu olduğunu raporlar / reports what works
```

İsteğe bağlı ekler / optional extras:

| Ek / extra | Ne sağlar / what it adds |
|---|---|
| `pip install -e ".[mesh]"` | Gmsh + meshio: PDMS duvarlı ağlar (düz kanal bunlarsız da çalışır) |
| `pip install -e ".[imaging]"` | Napari görüntüleyici, `btrack` soy ağacı |
| `pip install -e ".[segmentation]"` | Cellpose / StarDist segmentasyon arka uçları |
| `pip install -e ".[movie]"` | MP4 yörünge animasyonu (`imageio-ffmpeg`) |
| `pip install -e ".[dev]"` | pytest, ruff, mypy |

**Ağır çözücüler ayrı imajlardadır** (`biosim-lab-openfoam`, `biosim-lab-elmer`)
ve ana imaja bağımlılık eklemezler. Kurulu değillerse platform şunu bildirir:

> *streaming devre dışı, analitik Rayleigh yaklaşımı kullanılıyor*

---

## Tarayıcıda dene / Try it in a browser

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py             # http://localhost:8501
```

Aynı dosya Streamlit Community Cloud'a olduğu gibi dağıtılabilir — depoyu
GitHub'a itin, giriş noktası olarak `streamlit_app.py` seçin. Adım adım anlatım:
[USER\_MANUAL.md §9](docs/USER_MANUAL.md#9-publishing-your-own-copy-on-streamlit).

`requirements.txt` yönetilen bir sunucuda **gerçekten çalışan her şeyi** içerir.
Depoda bilerek **`packages.txt` yoktur**: böyle bir dosyanın varlığı sunucuda
`apt-get update` çalıştırır ve temel imajdaki süresi dolmuş tek bir depo tüm
dağıtımı düşürür — üstelik yeni örnek hiç başlamadığı için eski süreç eski kodla
hizmet vermeye devam eder. Hiçbir bağımlılık sistem kütüphanesi gerektirmez;
OpenGL olmadan içe aktarılamayan tek paket Gmsh'tir ve yapılandırılmış ağ
şablonuna geri düşer. Bu küme, Streamlit Cloud'un çalıştırdığı platformun aynısı
olan `linux/amd64` konteynerinde **hiç apt paketi kurulmadan** doğrulanmıştır.

Üçü bilerek dışarıdadır ve `requirements.txt`'e eklemek işe yaramaz: **Napari**
(Qt ve ekran ister), **OpenFOAM/Elmer** (Python paketi değil, harici ikili
dosya), **Cellpose/StarDist** (varsayılan PyTorch tekerleği ~2.5 GB CUDA taşır;
CPU sürümü için `requirements.txt` sonundaki dört satırı açın). Uygulamanın
*Environment* sayfası bunları *kurulu*, *eksik ama eklenebilir* ve *burada
çalışamaz* diye ayırarak canlı raporlar.

![Streamlit arayüzü](assets/streamlit_saw_sorter.png)

Panoda **canlı görünüm** sekmesi hücreleri tek tek, gerçek yarıçaplarıyla,
kanal boyunca hareket ederken gösterir (oynat düğmesi ve zaman kaydırıcısı ile);
ölü hücreler içi boş gri işaretlerdir. **Kesit** sekmesi kanalı ölçekli keser,
**canlı sayım** çıkışta biriken sayacı, **hücre güvenliği** ise üç hasar
mekanizmasının yayımlanmış eşiklere olan marjını verir.

## Hızlı başlangıç / Quick start

```bash
biosim list                                # kurulu cihaz ve çözücüler
biosim init saw_sorter -o my_run.yaml      # çalışır bir yapılandırma yaz
biosim run my_run.yaml                     # çalıştır, results/ altına yaz
biosim dashboard my_run.yaml               # tarayıcıda etkileşimli pano
```

Python'dan / from Python:

```python
from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.plugin import get_instrument

cfg = ExperimentConfig.from_yaml("configs/ctc_vs_rbc.yaml")
instrument = get_instrument(cfg.instrument)(cfg)
result = instrument.run()

print(result.metrics["efficiency_percent"], result.metrics["purity_percent"])
result.table.head()          # hücre başına bir satır / one row per cell
instrument.dashboard()       # Panel görünümü / Panel view
```

Birimler yapılandırmada açıkça yazılır ve `pint` ile denetlenir; çekirdeğe
yalnızca SI `float` geçer:

```yaml
params:
  frequency: 6.632 MHz     # "20 um" yazsanız DimensionalityError alırsınız
  channel_width: 300 um
  flow_rate: 5 uL/min
```

---

## Cihazlar / Instruments

| Eklenti | Ticari karşılığı | Durum | Ne yapar |
|---|---|---|---|
| `saw_sorter` | Akustik hücre ayırıcı | **Tam** | CTC/kan hücresi ayrımı, üç kip: duran SAW (`ssaw`), eğik açılı SAW (`tassaw`), zaman-anahtarlamalı iki frekanslı hacim dalgası (`alternating_baw`). Gor'kov kuvveti, analitik + FEM, RK4/adaptif entegrasyon, makale metrikleri, parametre taraması, canlı pano, **iki makaleye karşı doğrulanmış** |
| `impedance_rtca` | xCELLigence RTCA | **Minimal çalışır** | Altın interdijital elektrot (Olthuis hücre sabiti) + Giaever–Keese hücre katmanı; isteğe bağlı elektro-kuasistatik FEM; Cell Index ve normalleştirilmiş CI, \|Z\|(f) spektrumu, 4PL IC50; gerçek RTCA CSV/XLSX (geniş/uzun format) + plaka düzeniyle ölçülmüş veride IC50 |
| `cell_counter` | Countess / Cellometer | **İskelet + demo** | Watershed segmentasyon, Neubauer geometrisiyle konsantrasyon, tripan mavisi canlılık |
| `cell_tracker` | Incucyte / CellTracker | **İskelet + demo** | trackpy bağlama, hız / persistans / MSD |

Çözücü arka uçları / solver back-ends:

| Çözücü | Tür | Durum |
|---|---|---|
| `builtin_helmholtz` | akustik | **Her zaman kurulu** (scikit-fem) |
| `builtin_electroquasistatic` | elektro | **Her zaman kurulu** |
| `builtin_stokes` | akış | **Her zaman kurulu** |
| `openfoam` | akustik akış (streaming) | Opsiyonel — Aşama 4, yalnızca iskelet + şablon |
| `elmer` | piezoelektrik IDT | Opsiyonel — Aşama 4, yalnızca iskelet + şablon |

---

## Çalışma kipleri / Operating modes

`saw_sorter` üç farklı cihazı tek arayüzden simüle eder; `mode` alanı seçer.
One instrument, three devices, selected by `mode`:

| `mode` | Cihaz / device | Mekanizma / mechanism | Örnek yapılandırma |
|---|---|---|---|
| `ssaw` | Duran SAW, düğüm düzlemleri akışa paralel | Hücre düğüme göç eder ve durur | [`configs/ctc_vs_rbc.yaml`](configs/ctc_vs_rbc.yaml) |
| `tassaw` | Eğik açılı SAW (Li 2015) | Düğüm düzlemleri akışa θ açılı; hücre çok sayıda düğüm–karın bölgesinden geçer, sapma birikir | [`benchmarks/benchmark_01_tassaw/config.yaml`](benchmarks/benchmark_01_tassaw/config.yaml) |
| `alternating_baw` | Tek piezoseramik, iki rezonans arasında zaman anahtarlaması (Zhang 2023) | 1 MHz: tek düğüm W/2; 3 MHz: W/6, W/2, 5W/6. Her fazın kendi genliği ve süresi; hücre başına en az iki çevrim zorunlu | [`benchmarks/benchmark_02_alternating_baw/config.yaml`](benchmarks/benchmark_02_alternating_baw/config.yaml) |

```yaml
params:
  mode: alternating_baw
  channel_width: 737 um
  sheath_ratio: 2.0              # 1:2 örnek:kılıf → y0 < W/3 bandı akıdan türetilir
  switching:
    integrator: rk4              # sabit adım, her anahtarlama anına denk gelir
    min_cycles: 2
    phases:
      - {frequency: 1 MHz, duration: 0.8 s, voltage_pp: 9 V,
         reference_energy_density: 41.06 J/m^3, reference_voltage_pp: 9 V,
         energy_source: "kalibrasyon — kaynağını yazın"}
      - {frequency: 3 MHz, duration: 1.4 s, voltage_pp: 110 V,
         reference_energy_density: 19.94 J/m^3, reference_voltage_pp: 110 V,
         energy_source: "kalibrasyon — kaynağını yazın"}
```

`field_model: analytic | fem` kipten bağımsızdır (FEM yalnızca `ssaw` için).
0.2 öncesi `mode: analytic/fem` yazımı hâlâ okunur. Sürüş RF gücü olarak da
verilebilir (`power_drive`: dBm + bir referans basınç). Metrikler makalelerin
kendi adlarıyla: `capture_efficiency_percent`, `contamination_rate_percent`,
`recovery_rate_percent`, `background_removal_percent`, `separation_distance_um`.

---

## Literatür doğrulaması / Literature benchmarks (Aşama 1B)

İki yayımlanmış CTC ayırıcısı referans vaka olarak yeniden üretilir. Tam rapor,
şekiller ve her sapmanın olası nedeni: **[benchmarks/REPORT.md](benchmarks/REPORT.md)**.
Two published CTC separators are reproduced as reference cases; full report above.

| | Li et al. 2015 — taSSAW | Zhang et al. 2023 — alternating BAW |
|---|---|---|
| DOI | [10.1073/pnas.1504484112](https://doi.org/10.1073/pnas.1504484112) | [10.3390/ijms24043338](https://doi.org/10.3390/ijms24043338) |
| Kalibre edilen tek şey | 35 dBm'deki basınç — belirtilen optimum eğime (5°) | iki modun E_ac'si — makalenin W/6 tasarım kuralı ve "1 s sonrası minimum" ifadesine |
| Yeniden üretilen | Fig. 2A/2B/S2/3'ün 8 eğiliminin 8'i; ΔY 598 µm (makale ~600); IDT optimumu 10 mm (makale 8–10) | Table 1 yakalama verimi: MCF7 96.5 / HCT116 93.5 / A549 96.0 % (makale 95.0 / 94.4 / 94.6); Fig. 2 eğilimleri; Fig. 8 mekanizması |
| Sapma (DEVIATION) | Table 1: model 37.5 dBm'de fazla sürüyor (WBC uzaklaştırma %70, makale ~%90) | PBMC kontaminasyonu %17–31 (makale ~%1.5): örnek akışı kenarı kararsız W/3 antinodunda |
| Bulgu | 9.9 / 7.3 µm boncuk ayrımı bu geometride ≥%97'ye ulaşmıyor (sonuç başka bir cihazdan) | **Makalenin temel iddiası yeniden üretilemedi** — bkz. bulgu 5 |

Kurallar: referans değerler yalnızca makale **metnindeki ve tablolarındaki**
açık sayılardır, grafikten değer okunmaz; nicel sapma test başarısızlığı değil
DEVIATION olarak raporlanır; makalenin belirttiği **eğilimler** testte
zorunludur (`tests/test_benchmarks.py`).

```bash
biosim benchmark all            # iki benchmark + REPORT.md (~12 dk, 8 çekirdek)
biosim benchmark 02 --quick     # hızlı duman testi
```

---

## Fizik / Physics

Her formülün kaynağı DOI ile birlikte docstring'de yazılıdır. Öne çıkanlar:

**Birincil akustik radyasyon kuvveti (Gor'kov)**

```
F_r(x) = -(π p₀² V_c κ_f / 2λ) · Φ · sin(2k(x − x_düğüm))

Φ = (5ρ_p − 2ρ_f)/(2ρ_p + ρ_f) − κ_p/κ_f
```

`doi:10.1039/c2lc21068a` (Bruus 2012). **Dikkat:** literatürde iki tanım
dolaşır ve 3 kat farklıdırlar; `contrast_factor()` klasik (yukarıdaki) değeri,
`bruus_phi()` ise `Φ/3` değerini döndürür. Karıştırmak bu alandaki en yaygın
hatadır ve `tests/test_acoustics.py` iki tanımın oranını sabitler.

**Stokes sürüklenme** `F_d = 6πμr(u_f − u_p)`, `doi:10.1017/CBO9780511800955`.
Geçerlilik (`Re_p ≪ 1`, `Stk ≪ 1`) her koşuda denetlenir ve ihlalde
`RegimeWarning` verilir.

**Dikdörtgen kanalda Poiseuille akışı** — Fourier serisi çözümü, sayısal
integrasyonla bağımsız olarak doğrulanır (`verify_flow_rate()` bağıl hata
< 1e-3) ve geniş kanal limitinde `u_max/u_ort → 3/2` verir.

**Giaever–Keese empedans modeli** `doi:10.1073/pnas.88.17.7896` — Bessel
fonksiyonlu tam çözüm, Ω·cm² birim sisteminde.

**İnterdijital elektrot** — iki eşit tarak, iki arayüz seri; hacim direnci
Olthuis hücre sabiti `K = 2/((N−1)L)·K(k)/K(k')`, `k = cos(πw/2(w+s))`
(`doi:10.1016/0925-4005(95)85053-8`). `field_model: fem` aynı elektrodu
elektro-kuasistatik Poisson ile çözer; FEM hücre sabitini %0.3 içinde verir.
10 kHz okuma frekansında toplu model FEM'e %0.4 yakındır (Cell Index modelden
bağımsız); 100 kHz – 1 MHz'de akım parmak kenarlarına yığıldığı için %6'ya
kadar düşük tahmin eder.

**Hacim dalgası rezonansı (BAW)** — `p = p_a cos(nπy/W)`, kuvvet
`F = 4πΦ_B a³ k_n E_ac sin(2k_n y)` (`doi:10.1039/c2lc21068a`); kapalı form
yörünge `tan(k_n y) = tan(k_n y₀)·e^{t/τ}` (Barnkob 2010, `doi:10.1039/b920376a`);
RK4 entegratörü buna karşı < 1 nm doğrulanır. Aynı yasa, yukarıdaki formülün
`λ = 2W/n` ve orijini duvara taşınmış hâlidir.

**Sızıntılı SAW sınır koşulu** — taban yüzeyinde `v_n = -iω u₀ sin(k_SAW(x−x₀))`;
Rayleigh kırılım açısı (`θ_R = asin(c_f/c_SAW) ≈ 22°`) elle dayatılmaz,
Helmholtz çözümünden **kendiliğinden çıkar**.

---

## Bilmeniz gereken beş sonuç / Five findings you should know

Bunlar geliştirme sırasında ortaya çıktı ve tasarımınızı doğrudan etkiler.

### 1. SSAW'da düğüm aralığını λ_SAW belirler, λ_su değil

Duran yüzey akustik dalgası cihazında basınç düğümleri **λ_SAW/2** aralıklıdır
(`doi:10.1039/b910595f`), çünkü sıvıdaki alan tabanın periyodikliğini devralır.
128° YX LiNbO₃ üzerinde 20 MHz için λ_SAW = 199 µm → düğüm aralığı 99.5 µm.

**Sonuç:** şartnamedeki 20 MHz + 300 µm kanal kombinasyonu **tek düğümlü
değildir** — kanala üç düğüm sığar ve iki çıkışlı ayrım çalışmaz. Platform bunu
`RegimeWarning` ile bildirir ve doğru frekansı önerir:

```
f = c_SAW / (2·W) = 3979 / (2·300e-6) = 6.632 MHz
```

Varsayılan senaryo (`configs/ctc_vs_rbc.yaml`) 6.632 MHz kullanır ve **%100
geri kazanım, %94 saflık** verir. Şartnamedeki nokta
`configs/saw_sorter_20mhz_spec.yaml` içinde korunmuştur — çalıştırın, uyarıyı
ve üç modlu çıkış histogramını görün.

### 2. SSAW'da dipol terimi (k_x/k_f)² ile küçülür

Ders kitabı kuvvet yasası `k_x = k_f = ω/c` varsayar. SSAW'da bu **doğru
değildir**: yanal dalga sayısını taban dayatır (`k_x = 2π/λ_SAW`), sıvı ise
Helmholtz denklemine uyar, dolayısıyla alanın bir de dikey bileşeni vardır
(`k_y = √(k_f² − k_x²)` — tam olarak Rayleigh açısındaki kırılım).

Gor'kov potansiyeline koyunca **monopol terimi değişmez, dipol terimi
`(k_x/k_f)²` ile ölçeklenir**. `effective_contrast_factor()` bunu uygular ve
`k_x = k_f` limitinde klasik ifadeyi birebir geri verir. MCF-7 için 6.632
MHz'de: Φ = 0.2371 → Φ_etkin = 0.1787 (%25 azalma). Doğrulama testi bu terimi
tam bir Helmholtz çözümüne karşı 1e-3 bağıl RMS ile eşleştirir.

### 3. Sıcaklık her şeyi değiştirir ve önceden sessizce yok sayılıyordu

Suyun viskozitesi 25 °C'de 0.890, 37 °C'de 0.691 mPa·s'tir — **%22 düşüş**.
Akustoforetik hız `F/(6πμr)` olduğundan hücreler inkübatörde tezgâhtakinden
**%25 daha hızlı** göç eder. Tezgâhta ayarlanıp inkübatörde çalıştırılan bir
cihaz aynı cihaz değildir.

Yönü de sezgiye aykırıdır: ısıtmak saflığı **düşürür**, çünkü arka plan
popülasyonu da hızlanır ve daha fazlası toplama çıkışına ulaşır.

| Sıcaklık | Viskozite | Göç hızı | Referans koşuda saflık |
|---|---|---|---|
| 4 °C | 1.568 mPa·s | 0.55× | %97.6 |
| 25 °C | 0.890 mPa·s | 1.00× | %95.2 |
| 37 °C | 0.691 mPa·s | 1.25× | %93.0 |

Korelasyonlar Kell (1975), Marczak (1997) ve Kestin ve ark. (1978)'dendir ve
üçü de 25 °C'de kütüphane değerlerini dört anlamlı basamağa kadar geri verir.
Ayrıca suyun 4 °C yoğunluk anomalisi testle doğrulanır.

### 4. 1-B analitik model en iyi durumdur, FEM gerçekçi olandır

Kapalı form kuvveti **her yükseklikte taban düzlemi genliğiyle** uygular. Gerçek
sızıntılı SAW alanı ise yükseklikle zayıflar: 50 µm kanalda yanal kuvvet tavanda
taban değerinin ~%19'una düşer, kanal ortalaması **%53**'tür.

Bu yüzden analitik mod ayrım başarımını **iyimser** tahmin eder (bkz.
`examples/04_validate_analytic_vs_fem.py`, sayısal karşılaştırmayı basar).
Kuvvet profilinin biçimi %13 RMS içinde uyuşur; farkı yaratan yükseklik
zayıflamasıdır.

İlgili bir tuzak: FEM alanının **dikey** Gor'kov bileşeni varsayılan olarak
kapalıdır. Bu 2-B kesit modelinde onu dengeleyecek bir kaldırma kuvveti yoktur,
bu yüzden açılırsa hücreler eksenel hızın sıfır olduğu duvara yığılır ve
çıkıştan hiç geçmezler. `enable_vertical_arf: true` yaparsanız metriklerdeki
`all_cells_exited` alanını kontrol edin.

### 5. Sıkıştırılabilirlik farkı, benzer boyutlu CTC'yi PBMC'den ayırmaz — tersine

Gor'kov teorisinde **daha sıkıştırılabilir** bir hücrenin monopol katsayısı
`f₁ = 1 − κ_p/κ_f` ve dolayısıyla kontrast faktörü **küçüktür**; düğüme daha
yavaş gider. Zhang et al. (2023) Şekil 1'de kanser hücrelerini PBMC'lerden
*daha* sıkıştırılabilir ölçer (~4.3 vs ~4.0 ×10⁻¹⁰ Pa⁻¹) ve yine de "benzer
boyutlu CTC'ler ayrıştırıldı" der. Karşıt-olgu senaryosu (CTC 12 µm, PBMC
10.5 µm, eşit yoğunluk):

| Kol | En iyi Youden J |
|---|---|
| yalnız boyut | %36 — ayrışmıyor |
| + ölçülen sıkıştırılabilirlik (CTC daha yumuşak) | **%1** — daha kötü |
| + ters sıkıştırılabilirlik (CTC daha sert) | %70 — daha iyi |

Yani sıkıştırılabilirlik ancak CTC'ler PBMC'lerden **daha sert** olsaydı
yardım ederdi. Makalenin iddiası birincil radyasyon kuvvetiyle açıklanamıyor;
ölçülmemiş bir yoğunluk farkı, akustik streaming ya da "sıkıştırılabilirlik"
ölçümünün farklı bir anlamı gerekir. Bu bir **bulgudur**, model hatası değil:
`tests/test_benchmarks.py::test_counterfactual_follows_gorkov` modelin bu yönü
tutarlı biçimde verdiğini zorunlu kılar.

---

## Örnekler / Examples

```bash
python examples/01_single_cell.py               # tek hücre kuvvet dengesi
python examples/02_ctc_vs_rbc.py                # MCF-7 vs eritrosit + şekiller
python examples/03_parameter_sweep.py --quick   # frekans × voltaj × debi taraması
python examples/04_validate_analytic_vs_fem.py  # analitik ↔ FEM doğrulaması
python examples/05_impedance_rtca.py            # IDE + FEM, Cell Index, Nyquist/Bode, IC50
python examples/06_imaging_demo.py              # sayım + takip, sentetik veriyle
python examples/07_assumption_sensitivity.py   # hangi kaynaksız sayı sonucu değiştiriyor?
python examples/08_pipeline_sort_count_track.py # üç cihaz tek iş akışı + hata bütçesi
python examples/09_two_outlet_split.py          # iki çıkışlı ayırma: ayırıcı nereye?
python examples/10_tilted_angle_ssaw.py         # eğik açılı SSAW: tutulma sınırı
python examples/11_ctc_from_blood.py            # kandan CTC: gerçek oranlar + yayımlanmış protokol
python examples/12_alternating_baw.py           # iki frekanslı BAW: zaman izleri, pencere, RK4 vs LSODA
```

Şekiller `assets/` altına hem etkileşimli HTML hem PNG olarak yazılır.

### Pano / Dashboard

`biosim dashboard configs/ctc_vs_rbc.yaml` (veya `docker compose up biosim`) →
<http://localhost:5006>

![SAW sorter dashboard](assets/dashboard_saw_sorter.png)

Kaydırıcılar simülasyonu **canlı** yeniden çalıştırır: sürüş voltajını 15 → 6.5
Vpp indirdiğinizde geri kazanım %100'den %1'e düşer, saflık %95.5'ten %100'e
çıkar — az sayıda ama saf hücre. Sekmeler yörüngeler, çıkış histogramı, kuvvet
profili, boyut dağılımı ve popülasyon tablosunu gösterir.

`examples/02_ctc_vs_rbc.py` çıktısı:

```
  recovery (efficiency)  100.0 %
  purity                  94.3 %
  enrichment              1.89 x

  population     n  collected   |dx| (um)  to node (um)
  -----------------------------------------------------
  mcf7         400     100.0%       116.1           4.0
  rbc          400       6.0%        31.3          97.1
```

---

## Komut satırı / CLI

| Komut | Ne yapar |
|---|---|
| `biosim list` | Kurulu cihazları ve çözücü arka uçlarını listeler |
| `biosim doctor` | Ortamı tarar: hangi bağımlılık var, hangi yetenek kapalı |
| `biosim init <cihaz> -o c.yaml` | Çalışan örnek yapılandırma yazar |
| `biosim run c.yaml` | Deneyi çalıştırır, `.nc` + `.parquet` + `.csv` yazar |
| `biosim sweep c.yaml -p 'voltage_pp=5 V,15 V'` | Parametre taraması |
| `biosim dashboard c.yaml` | Panel panosunu tarayıcıda açar |
| `biosim materials` | Malzeme kütüphanesini DOI / ASSUMPTION etiketleriyle basar |
| `biosim benchmark [01\|02\|all] [--quick]` | Literatür benchmark'larını çalıştırır, `benchmarks/REPORT.md`'yi yeniden üretir (kaynak ağacından) |

---

## Docker

```bash
docker compose up biosim            # pano: http://localhost:5006
docker compose run --rm tests       # test paketi
docker compose run --rm examples    # tüm örnekleri çalıştır, assets/ üret
```

Ağır çözücüler ayrı profillerdedir ve ana imaja hiçbir şey eklemezler:

```bash
docker compose --profile solvers build openfoam elmer
```

---

## Kaynaklar ve varsayımlar / Provenance and assumptions

Malzeme kütüphanesindeki **her sayı** ya bir DOI'ye dayanır ya da açıkça
`ASSUMPTION` etiketlidir ve neden varsayıldığı yazılıdır. Üçüncü bir kategori
yoktur; `tests/test_materials.py` bunu zorunlu kılar.

```bash
biosim materials                 # yalnızca varsayımlar
biosim materials --all           # her değer, kaynağıyla
```

Bir çalışmanın yöntem bölümünde açıklanması gereken liste tam olarak budur.
Programatik erişim:

```python
from biosim_lab.core.materials import audit
for row in audit():
    print(row["material"], row["property"], row["provenance"])
```

En büyük tek varsayım: IDT sürüş voltajını akustik basınca çeviren doğrusal
kalibrasyon (`PRESSURE_PER_VOLT_ASSUMPTION`, 15 Vpp → 0.45 MPa). Benchmark'lar
bunu kullanmaz: RF gücünden (`power_drive`) ya da doğrudan enerji
yoğunluğundan sürer ve referans değeri **CALIBRATED** etiketiyle, hangi
belirtilmiş kurala dayandığını yazarak verir. Popülasyon başına hücre özelliği
geçersiz kılmaları (`diameter`, `compressibility`, …) da kaynak (`override_source`)
belirtilmeden kabul edilmez. Bunu kaldırmak
Aşama 4'ün (Elmer piezoelektrik çözümü) işidir; o zamana kadar kendi çipiniz
için `pressure_amplitude` değerini ölçüp doğrudan verin.

---

## Belgeler / Documentation

| Belge | İçerik |
|---|---|
| [docs/USER\_MANUAL.tr.md](docs/USER_MANUAL.tr.md) | **Kullanım kılavuzu** — kurulum, web arayüzü, CLI, Python API, tam yapılandırma referansı, kendi verinizle çalışma, Streamlit'e dağıtım, sorun giderme |
| [docs/USER\_MANUAL.md](docs/USER_MANUAL.md) | The same manual, in English |
| [docs/explainer/](docs/explainer/) | Uzman olmayanlar için 31 sayfalık PDF anlatım (**Türkçe** ve İngilizce) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Katmanlar, sözleşmeler, veri akışı, sınırlamalar |
| [docs/physics.md](docs/physics.md) | Her formül, kaynağı ve geçerlilik sınırı |
| [docs/validation.md](docs/validation.md) | Ne, neye karşı, hangi toleransla doğrulandı |
| [benchmarks/REPORT.md](benchmarks/REPORT.md) | Aşama 1B: iki makaleye karşı sonuç / makale / sapma tabloları ve şekiller |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Yeni cihaz eklemek — 10 adım |

## Uzman olmayanlar için / For non-experts

Biyolojiyi, fiziği, matematiği ve hesaplamayı sıfırdan anlatan 31 sayfalık bir
belge: [`docs/explainer/biosim-lab-nasil-calisir.pdf`](docs/explainer/). Beşte
biri modelin **yanlış yaptığı** şeylere ayrılmıştır, çünkü okuyucunun geri
kalanına güvenip güvenmeyeceğine karar vermesi için gereken kısım odur.
İngilizce sürümü de aynı klasörde (`biosim-lab-explained.pdf`, 29 sayfa).

## Katkı / Contributing

Yeni bir cihaz eklemek **10 adımdır** ve çekirdekte hiçbir değişiklik
gerektirmez — bkz. [CONTRIBUTING.md](CONTRIBUTING.md). Mimari için
[ARCHITECTURE.md](ARCHITECTURE.md).

```bash
pip install -e ".[dev]"
pytest                    # 208 test, opsiyonel arka uç olmadan geçer
ruff check biosim_lab
```

Her itme ve her PR'da GitHub Actions altı iş çalıştırır
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

| İş | Ne denetler |
|---|---|
| `test` | Python 3.11 ve 3.12'de lint + tüm test paketi, kapsam raporuyla |
| `minimal` | **Hiçbir opsiyonel arka uç kurulu değilken** paketin yine geçmesi — mimarinin temel iddiası |
| `figures` | Örnekleri çalıştırır ve üretilen hiçbir şeklin boş olmadığını doğrular |
| `package` | Wheel derlenir, temiz bir ortama kurulur, entry point'ler çözülür |
| `streamlit` | `requirements.txt` ile uygulama import edilir ve dört cihaz çıplak checkout'ta bulunur |
| `types` | mypy (bilgilendirme amaçlı; şu an 36 bilinen hata) |

## Atıf / Citation

Bkz. [CITATION.cff](CITATION.cff). Literatür benchmark'larının dayandığı makaleler /
the papers the benchmarks reproduce:

- Li P, Mao Z, Peng Z, et al. (2015) Acoustic separation of circulating tumor cells.
  *PNAS* 112(16):4970–4975. doi:[10.1073/pnas.1504484112](https://doi.org/10.1073/pnas.1504484112)
- Zhang Y, Zhang Z, Zheng D, Huang T, Fu Q, Liu Y (2023) Label-free separation of
  circulating tumor cells and clusters by alternating frequency acoustic field in a
  microfluidic chip. *Int J Mol Sci* 24(4):3338. doi:[10.3390/ijms24043338](https://doi.org/10.3390/ijms24043338)

## Lisans / License

MIT — bkz. [LICENSE](LICENSE).
