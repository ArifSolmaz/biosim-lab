# biosim-lab — Kullanım Kılavuzu

Sürüm 0.1.0 · MIT lisansı

> İngilizce sürüm: [`USER_MANUAL.md`](USER_MANUAL.md). İki belge aynı içeriği
> taşır; biri güncellenirse öbürü de güncellenmelidir.

---

## İçindekiler

1. [Tek sayfada bu nedir](#1-tek-sayfada-bu-nedir)
2. [Üç kullanım yolu](#2-üç-kullanım-yolu)
3. [Yerel kurulum](#3-yerel-kurulum)
4. [Web arayüzü, sayfa sayfa](#4-web-arayüzü-sayfa-sayfa)
5. [Komut satırı](#5-komut-satırı)
6. [Python API'si](#6-python-apisi)
7. [Yapılandırma dosyası referansı](#7-yapılandırma-dosyası-referansı)
8. [Kendi verinizle çalışmak](#8-kendi-verinizle-çalışmak)
9. [Streamlit'te kendi kopyanızı yayımlamak](#9-streamlitte-kendi-kopyanızı-yayımlamak)
10. [Docker](#10-docker)
11. [Sonuçları dürüstçe okumak](#11-sonuçları-dürüstçe-okumak)
12. [Sorun giderme](#12-sorun-giderme)
13. [Genişletmek](#13-genişletmek)

---

## 1. Tek sayfada bu nedir

biosim-lab, dört tür laboratuvar cihazının ölçtüğü şeyi yalnızca açık kaynak
Python ile simüle eder. Aynı analizleri gerçek cihaz verisiyle de çalıştırır.

| Cihaz | Ticari karşılığı | Yanıtladığı soru | Durum |
|---|---|---|---|
| `saw_sorter` | akustik hücre ayırıcılar | Kandan tümör hücresi çekebilir miyim, sonuç ne kadar saf? | tam |
| `impedance_rtca` | xCELLigence RTCA | Hücreler ne kadar hızlı büyüyor, hangi ilaç dozu yarısını öldürüyor? | minimal ama çalışır |
| `cell_counter` | Countess, Cellometer | Mililitrede kaç hücre var, kaçı canlı? | iskelet + demo |
| `cell_tracker` | Incucyte | Ne kadar hızlı sürünüyorlar, bir yöne mi gidiyorlar? | iskelet + demo |

Hepsi tek bir çekirdeği paylaşır: doğrulanmış bir yapılandırma, bir çözücü, bir
sonuç kabı ve bir çizim katmanı. Beşinci bir cihaz eklemek o çekirdekte hiçbir
değişiklik gerektirmez — bkz. [§13](#13-genişletmek).

**Başka tek bir şey okuyacaksanız**, uzman olmayanlar için yazılmış 31 sayfalık
[`docs/explainer/biosim-lab-nasil-calisir.pdf`](explainer/) belgesini okuyun.
Beşte biri modelin neyi yanlış yaptığına ayrılmıştır.

---

## 2. Üç kullanım yolu

| Yol | Ne için iyi | Gerekenler |
|---|---|---|
| **Web arayüzü** | keşfetmek, ders anlatmak, birine göstermek | tarayıcı (barındırılan) ya da `pip install -r requirements.txt` |
| **Komut satırı** | yinelenebilir koşular, toplu taramalar, sonuç kaydetmek | yerel kurulum |
| **Python API'si** | kendi analiziniz, defterler, yeni cihazlar | yerel kurulum |

Üçü de aynı koddur. Web arayüzü, komut satırının çağırdığı fonksiyonların tam
olarak aynılarını çağırır; ekrandaki hiçbir şey önceden pişirilmemiştir.

---

## 3. Yerel kurulum

**Python 3.11 veya üstü** gerekir.

```bash
git clone https://github.com/<hesabınız>/biosim-lab.git
cd biosim-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[mesh]"
biosim doctor                      # neyin çalıştığını doğrular
```

### Hangi kurulumu istiyorum?

| Komut | Getirdiği |
|---|---|
| `pip install -e .` | Gmsh ağ üretimi hariç her şey |
| `pip install -e ".[mesh]"` | **önerilen** — Gmsh + meshio ekler, yalnızca PDMS duvar katmanı için gerekir |
| `pip install -r requirements.txt` | web uygulamasının kümesi (yönetilen bir sunucuda çalışan her şey) |
| `pip install -e ".[imaging]"` | Napari görüntüleyici ve `btrack` soy ağaçları |
| `pip install -e ".[segmentation]"` | Cellpose / StarDist bölütleme arka uçları |
| `pip install -e ".[dev]"` | pytest, ruff, mypy |

### Kurulumu denetleyin

```bash
pytest                     # 230 test, ~15 s, hepsi geçmeli
biosim list                # 4 cihaz, 5 çözücü arka ucu
```

`pytest`'in **hiçbir opsiyonel arka uç kurulu değilken** geçmesi bir rastlantı
değil, bilinçli bir tasarım güvencesidir ve sürekli tümleştirmede ayrı bir iş
olarak denetlenir.

### Bir şey eksikse

`biosim doctor` asla başarısız olmaz — rapor verir. Eksik her bileşenin yanında
neyi açacağı yazar, böylece ona ihtiyacınız olup olmadığına karar verebilirsiniz.

---

## 4. Web arayüzü, sayfa sayfa

Yerelde çalıştırmak için:

```bash
streamlit run streamlit_app.py
```

<http://localhost:8501> adresinde açılır.

### Genel bakış

Yönlendirme, artı bir sayıya güvenmeden önce bilinmesi gereken üç şey. Kurulumu
sizin için başkası yaptıysa buradan başlayın.

### SAW hücre ayırıcı

Amiral gemisi. Kenar çubuğundaki her denetim simülasyonu yeniden çalıştırır.

**Fiziği değiştiren denetimler**

| Denetim | Etkisi | Dikkat |
|---|---|---|
| Kanal genişliği | hangi frekansın tek düğüm vereceğini belirler | ipucu balonu, o genişlik için tek düğümlü frekansı yazar |
| Frekans | düğüm aralığını belirler (`λ_SAW/2`) | kanalda birden çok düğüm olursa iki çıkışlı ayrım çalışmaz |
| Sürüş voltajı | kuvvet **voltajın karesiyle** ölçeklenir | açık ara en etkili düğme |
| Debi | hücrelerin alanda ne kadar kalacağını belirler | artırın, geri kazanım her zaman düşer |
| Etkin uzunluk | debiyle ters yönde aynı etki | gerçek çipte IDT açıklığıdır |
| **Sıcaklık** | viskozite, ve göç hızı onunla ters orantılı | 25 → 37 °C her hücreyi ~%25 hızlandırır ve bu saflığı **düşürür** |
| **Örnek canlılığı** | girişte kaçının canlı olduğu | taze bir süspansiyon %90–97 canlıdır, %100 değil |
| Kuvvet alanı: analitik / FEM | kapalı form mu, dalga denklemini çözmek mi | FEM daha yavaş ve **daha az iyimserdir** — bkz. [§11](#11-sonuçları-dürüstçe-okumak) |
| **Tekrarlar** | aynı cihazı taze hücre örnekleriyle yeniden koşar | 1 yalnızca sayım hatasını verir; daha fazlası örnekler arası yayılımı da ölçer |

**Sayıları okumak**

- **Geri kazanım** — hedef hücrelerin kaçta kaçı toplama çıkışına ulaştı.
- **Saflık** — toplananların kaçta kaçı hedef.
- **Canlı saflık** — toplanan *canlı* hücrelerin kaçta kaçı hedef. Toplanan ölü
  bir hücre sonraki aşamaya yaramaz, bu yüzden genellikle aktarılacak dürüst
  rakam budur.
- **Zenginleşme** — saflığın girdi oranına bölümü. Nadir hücre tahlilinin
  gerçekten yargılandığı sayı. Geri kazanımla saflık takas edilir; zenginleşme
  bu takas konusunda yalan söylemez.
- **Çıkış canlılığı** — yanındaki değişim, cihazın kendisinin kaç puana mal
  olduğunu gösterir. Bu rejimde sıfır olmalıdır; değilse *Hücre güvenliği*
  sekmesine bakıp hangi eşiğin aşıldığını görün.

**Sekmeler**

- *Canlı görünüm* — **her hücre hareket eden bir nokta olarak**, yukarıdan
  görülür; oynat düğmesi ve zaman kaydırıcısı vardır. İşaret boyutu gerçek
  yarıçapı izler; içi boş gri işaretler ölü hücrelerdir. Bu, çipin üzerindeki bir
  mikroskobun size vereceği görüntüdür.
- *Kesit* — kanal çıkışta enine kesilmiş, hücreler **ölçekli** çizilmiş. Bütün
  ayrımı sürükleyen boyut farkı doğrudan görünür.
- *Canlı sayım* — her çıkışta biriken sayaç, bir cihazın göstereceği gibi.
  Eğimi, saniyedeki hücre olarak verimdir.
- *Hücre güvenliği* — üç hasar mekanizması, yayımlanmış eşikleri ve çalışma
  noktasının her birine olan marjı.
- *Belirsizlik* — bir sayıya ne kadar güvenileceği; iki ayrı belirsizlik yan yana.
- *Yörüngeler*, *Çıkış histogramı*, *Kuvvet profili*, *Boyut dağılımı*,
  *Popülasyon başına*, *Veri* (CSV / YAML / Parquet / NetCDF indirmeleri).

### Empedans (RTCA)

96 kuyulu bir plakayı simüle eder: hücreler yapışır, büyür, doz alır ve bitiş
noktasından bir IC50 uydurulur. Ya da gerçek bir RTCA dışa aktarımını (CSV/XLSX)
yükleyin, onu çözümler.

Uydurulan IC50'nin ekilen değerden neden farklı olduğunu açıklayan mavi not bir
özür değil; gerçek bitiş noktası deneylerinin gösterdiği maruziyet süresi
etkisidir.

### Hücre sayıcı

Doğru yanıtı bilinen sentetik bir görüş alanını bölütler, böylece bölütleme
hayranlıkla seyredilmek yerine *puanlanabilir*. Ham görüntüyü saptanan
çevritlerin yanında gösterir.

Poisson sayım hatası (`1/√N`) belirgin biçimde gösterilir, çünkü genellikle
insanların ölçmeye çalıştığı farktan büyüktür.

### Hücre takipçisi

Her kareyi bölütler, saptamaları bağlar ve hızı, yönelim persistansını ve MSD
üsteli α'yı bildirir (1 = rastgele dolaşma, 2 = bir yere yürüme).

Önemli olan kaydırıcı *Arama menzilidir*: çok küçükse izler parçalanır, çok
büyükse kimlikler takas olur. Gerçek adım boyuna yaklaştığınızda uygulama uyarır.

### Malzeme kaynakları

65 fiziksel sabitin tamamı kaynağıyla: 27'si yayımlanmış DOI'li, 38'i gerekçesi
yazılı işaretli varsayım. CSV olarak indirilebilir. Bir yöntem bölümünün
açıklaması gereken liste budur.

### Ortam

Neyin kurulu olduğu, neyin olmadığı ve eksik her bileşenin neyi açacağı. Üç
başlık altında ayrılır: *kurulu*, *eksik ama eklenebilir*, *burada çalışamaz* —
hangi tür eksiklikle karşı karşıya olduğunuzu tahmin etmeniz gerekmez.

---

## 5. Komut satırı

| Komut | Ne yapar |
|---|---|
| `biosim list` | kurulu cihazları ve çözücü arka uçlarını listeler |
| `biosim doctor` | tam ortam tanısı |
| `biosim init <cihaz> -o run.yaml` | çalışan örnek yapılandırma yazar |
| `biosim run run.yaml` | çalıştırır, `.nc` + `.parquet` + `_metrics.csv` kaydeder |
| `biosim run run.yaml -n 5` | beş tekrarla koşar, yayılımı da bildirir |
| `biosim sweep run.yaml -p 'AD=v1,v2'` | parametre taraması |
| `biosim dashboard run.yaml` | Panel panosunu sunar |
| `biosim materials` | varsayım listesi |
| `biosim materials --all` | kaynağıyla birlikte her değer |

### Tam bir oturum

```bash
biosim init saw_sorter -o kosum.yaml
$EDITOR kosum.yaml                      # frekansı, hücre sayısını, ne isterseniz değiştirin
biosim run kosum.yaml -n 5
```

Çıktı, nokta tahminlerinin ardından tekrarlar arası yayılımı da içerir:

```
      across 5 replicates (different cell samples)
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ metric                ┃ mean ┃    std ┃      95 % CI ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ efficiency_percent    │ 99.9 │  0.298 │  [99.5, 100] │
│ purity_percent        │ 94.1 │   1.27 │ [92.5, 95.7] │
└───────────────────────┴──────┴────────┴──────────────┘
```

### Taramalar

Değerler birim taşıyabilir ve `-p` yinelenerek ızgara çok boyutlu yapılabilir:

```bash
biosim sweep kosum.yaml \
  -p 'voltage_pp=5 V,10 V,15 V,25 V' \
  -p 'flow_rate=5 uL/min,15 uL/min,40 uL/min' \
  -o taramalar/
```

> **Bir taramayı frekans boyunca ortalamayın.** Frekans basınç düğümlerinin
> *sayısını*, yani çalışma rejimini değiştirir. `examples/03_parameter_sweep.py`
> tam bu nedenle frekans başına bir ısı haritası üretir, hepsinin ortalamasını
> değil.

---

## 6. Python API'si

```python
from biosim_lab.core.config import ExperimentConfig
from biosim_lab.registry import installed_instruments

cfg = ExperimentConfig.from_yaml("configs/ctc_vs_rbc.yaml")
Cihaz = installed_instruments()[cfg.instrument]
sonuc = Cihaz(cfg).run()

sonuc.metrics["efficiency_percent"]   # skalerler
sonuc.metrics["efficiency_ci_low"]    # sayım hatası, %95 Wilson aralığı
sonuc.table.head()                    # hücre başına bir satır
sonuc.fields                          # yörünge/alan xarray Dataset'i
```

Ya da yapılandırma dosyası olmadan doğrudan simülasyonu sürün:

```python
from biosim_lab.instruments.saw_sorter.simulate import (
    SAWSorterParams, SAWSorterSimulation, replicate_sorting,
)

params = SAWSorterParams(
    frequency="6.632 MHz",        # birim dizeleri pint ile doğrulanır
    voltage_pp="15 V",
    temperature="37 degC",
    channel_width="300 um",
    flow_rate="5 uL/min",
    populations=[
        {"cell_type": "mcf7", "count": 300, "target": True},
        {"cell_type": "rbc",  "count": 300, "target": False},
    ],
)
sonuc = SAWSorterSimulation(params).run()
tekrarlar = replicate_sorting(params, n_replicates=8)   # örnekler arası yayılım
```

### Fizik fonksiyonlarını tek başına kullanmak

```python
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    contrast_factor, primary_radiation_force_1d, saw_wavelength,
)
from biosim_lab.core.materials import MCF7, WATER

phi = contrast_factor(MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa)   # 0.2371
lam = saw_wavelength(6.632e6, 3979.0)                                  # 600 µm
```

Her fizik fonksiyonu **birimsiz SI `float`** alır ve SI döndürür. Birimler
yapılandırma sınırında denetlenir, sayısal çekirdeğin içinde değil.

---

## 7. Yapılandırma dosyası referansı

Her koşu tek bir YAML dosyasıdır:

```yaml
name: deneyim               # çıktı dosya adlarında kullanılır
instrument: saw_sorter      # hangi eklenti
seed: 12345                 # yinelenebilirlik; rastgele için null
output_dir: results         # göreli yollar bu dosyanın yanına çözülür
description: >-
  Serbest metin, sonuç meta verisine taşınır.
params:                     # cihazın kendi şemasıyla doğrulanır
  frequency: 6.632 MHz
  ...
```

Birim taşıyan değerler dize olarak yazılır ve `pint` ile denetlenir.
`frequency: 300 um` yazmak koşmak yerine `DimensionalityError` verir.

### `saw_sorter`

| Parametre | Varsayılan | Anlamı |
|---|---|---|
| `frequency` | `6.632 MHz` | IDT sürüş frekansı |
| `voltage_pp` | `15 V` | sürüş voltajı; `pressure_amplitude` verilirse yok sayılır |
| `pressure_amplitude` | `null` | ölçülmüş p₀ — **elinizde varsa bunu yeğleyin** |
| `temperature` | `298.15 K` | `"37 degC"` kabul eder. **Tezgâhla inkübatör arasında göç hızını ~%25 değiştirir** |
| `rf_power` | `null` | uygulanan RF gücü [W], yalnızca işaretli dönüştürücü ısınma tahmini için |
| `inlet_viability` | `0.95` | cihaza girmeden önce zaten canlı olan kesir |
| `track_viability` | `true` | hücre başına termal, kayma ve kavitasyon hasarını hesapla |
| `substrate` | `linbo3_128yx` | SAW hızını belirler |
| `channel_width` / `_height` / `_length` | `300 um` / `50 um` / `2 mm` | `channel_length` = IDT açıklığı |
| `flow_rate` | `5 uL/min` | şırınga pompasının denetlediği büyüklük |
| `fluid` | `water` | `water`, `pbs`, `dmem` |
| `populations` | MCF-7 + RBC | `{cell_type, count, target}` listesi |
| `inlet` | `sheath_sides` | `sheath_sides`, `uniform`, `centre` |
| `collection_fraction` | `0.333` | merkezi çıkış genişliği / kanal genişliği |
| `mode` | `analytic` | `analytic` ya da `fem` |
| `integration` | `overdamped` | `overdamped` ya da `inertial` |
| `enable_vertical_arf` | `false` | FEM dikey kuvveti — bkz. [§11](#11-sonuçları-dürüstçe-okumak) |
| `enable_gravity` / `_wall_repulsion` / `_secondary_bjerknes` | `false` | isteğe bağlı ikincil etkiler |
| `fem_resolution` / `fem_grid` | `40` / `[241, 41]` | ağ ve örnekleme ızgarası |
| `n_time_samples` / `seed` | `101` / `12345` | |

Hücre türleri: `mcf7`, `hela`, `a549`, `rbc`, `wbc`, `platelet`, `ps_bead`,
`lipid`.

### `impedance_rtca`

| Parametre | Varsayılan | Anlamı |
|---|---|---|
| `frequency` | `10 kHz` | Hücre İndeksi okuma frekansı |
| `spectrum_frequencies` / `spectrum_range` | `60` / `[100, 1e7]` | `\|Z\|(f)` taraması |
| `n_wells` | `96` | `96` ya da `384` |
| `duration` / `n_timepoints` | `48 h` / `97` | |
| `doubling_time` / `lag_time` | `20 h` / `2 h` | büyüme ve yapışma gecikmesi |
| `seeding_coverage` / `max_coverage` | `0.05` / `0.95` | elektrot örtü sınırları |
| `treatment_time` | `24 h` | ilacın eklendiği an; yoksa `null` |
| `concentrations` / `replicates` | 8 nokta / `3` | doz serisi ve kuyu sayısı |
| `true_ic50` / `hill_slope` | `1.0` / `1.3` | sentetik plakayı üretmek için |
| `noise_cv` | `0.02` | ölçüm gürültüsü |
| `conductivity` / `junction_resistance` / `membrane_capacitance` | `1.4` / `2.0` / `1e-6` | elektriksel model |
| `electrode_area_cm2` / `cell_radius` / `gap_height` | `0.008` / `8 um` / `100 nm` | kabuk modeli geometrisi |
| `source_file` | `null` | gerçek RTCA dışa aktarımının yolu |

### `cell_counter`

| Parametre | Varsayılan | Anlamı |
|---|---|---|
| `source_image` | `null` | TIFF/PNG; yoksa sentetik üretir |
| `pixel_size` / `chamber_depth` | `0.65 um` / `100 um` | Neubauer standardı |
| `dilution_factor` | `2.0` | 1:1 tripan mavisi karışımı |
| `backend` | `classical` | `classical`, `cellpose`, `stardist` |
| `min_radius_px` | `4.0` | döküntü kesimi |
| `min_diameter_um` / `max_diameter_um` | `5` / `40` | boyut kapısı |
| `viability_threshold` | `null` | `null` parlaklıkta Otsu kullanır |
| `n_cells`, `dead_fraction`, `image_size`, `seed` | | sentetik demo denetimleri |

### `cell_tracker`

| Parametre | Varsayılan | Anlamı |
|---|---|---|
| `source_movie` | `null` | TIFF yığını ya da görüntü dizini |
| `pixel_size` / `frame_interval` | `0.65 um` / `10 min` | |
| `backend` | `classical` | |
| `search_range_px` | `12.0` | **en çok önem taşıyan parametre** |
| `memory_frames` / `min_track_length` | `2` / `5` | |
| `n_frames`, `n_cells`, `image_size`, `speed_px_per_frame`, `persistence`, `seed` | | sentetik demo denetimleri |

---

## 8. Kendi verinizle çalışmak

### Gerçek RTCA dışa aktarımları

```yaml
instrument: impedance_rtca
params:
  source_file: /yol/RTCA_export.csv
```

Okuyucu başlık satırının üstündeki metadata satırlarını atlar, farklı alan
sayısına sahip satırlara dayanıklıdır, ayracı sezer ve başlığı kuyu etiketlerini
(`A1` … `H12`) içeren ilk satırı bularak yerleştirir. CSV ve XLSX çalışır.

Ya da dosyayı web arayüzündeki yükleyiciye bırakın.

### Mikroskopi görüntüleri

```yaml
instrument: cell_counter
params:
  source_image: /yol/alan.tif
  pixel_size: 0.325 um        # SİZİN objektifiniz ve kameranız, varsayılan değil
  chamber_depth: 100 um
```

Takip için `source_movie` çok sayfalı bir TIFF **ya da numaralandırılmış
görüntülerden oluşan bir dizin** kabul eder.

> **`pixel_size` değerini doğru girin, yoksa her fiziksel sayı yanlış olur.**
> Pikseli metreye çevirir, dolayısıyla çapa, derişime ve hıza yayılır. Objektifin
> nominal büyütmesinden değil, bir sahne mikrometresinden alın.

### Ölçülmüş akustik basınç

Çipinizi kalibre ettiyseniz voltaj varsayımını tümüyle atlayın:

```yaml
params:
  voltage_pp: null
  pressure_amplitude: 0.38 MPa    # ölçülmüş, örneğin boncuk izleyerek
```

### Sıcaklık

```yaml
params:
  temperature: 37 degC        # ya da "310.15 K", ya da "98.6 degF"
```

| Sıcaklık | Viskozite | Göç hızı | Referans koşuda saflık |
|---|---|---|---|
| 4 °C | 1.568 mPa·s | 0.55× | %97.6 |
| 25 °C | 0.890 mPa·s | 1.00× | %95.2 |
| 37 °C | 0.691 mPa·s | 1.25× | %93.0 |

Yönüne dikkat: ısıtmak hücreleri hızlandırır, bu da saflığı **düşürür**, çünkü
arka plan popülasyonu da hızlanır ve daha fazlası toplama çıkışına ulaşır.
Soğutmak bir saflık düğmesidir.

---

## 9. Streamlit'te kendi kopyanızı yayımlamak

Depo, Streamlit Community Cloud'un ihtiyaç duyduğu her şeyi zaten içerir.

| Dosya | Rolü |
|---|---|
| `streamlit_app.py` | giriş noktası, depo kökünde |
| `requirements.txt` | Python bağımlılıkları — **yönetilen bir sunucuda çalışan tam küme** |
| `.streamlit/config.toml` | şekil paletiyle uyumlu tema, yükleme sınırı |
| `biosim_lab/app/` | uygulamanın kendisi: `app/pages/` altında sayfa başına bir modül |

Depoda bilerek **`packages.txt` yoktur** ve böyle bir dosya eklemek bu dağıtımı
düşürmenin en kolay yoludur.

### Neden `packages.txt` yok

Dosyanın varlığı, Streamlit Cloud'un pip'ten önce `apt-get update` çalıştırmasına
yol açar. Temel imaj, bu projenin denetiminde olmayan depoları taşır ve bunlardan
birinin süresinin dolması dağıtımı batırmaya yeter:

```
E: Release file for .../bullseye-security/InRelease is expired (invalid since 12h)
❗️ installer returned a non-zero exit code
❗️ Error during processing dependencies!
```

Sonuç, bir özelliği kaybetmekten daha kötüdür. Yeni örnek hiç başlamaz, bu yüzden
**önceki süreç hizmet vermeye devam eder** — yeni çekilmiş kaynağa karşı eski
bayt kodunu çalıştırarak. Az önce ittiğiniz düzeltme hiçbir şeyi değiştirmemiş
gibi görünür ve yığın izi eski satır numaralarıyla yeni kaynak metnini karıştırır.
Uygulama artık bu durumu saptayıp her sayfada bildirir (bkz. §12).

Burada hiçbir şey sistem kütüphanesine ihtiyaç duymaz. OpenGL olmadan içe
aktarılamayan tek bağımlılık Gmsh'tir ve o da tasarım gereği isteğe bağlıdır:
ağ oluşturma, yapılandırılmış düz kanal şablonuna geri düşer ve `biosim doctor`
onu eksik olarak raporlar. `requirements.txt` içindeki diğer her şey yardımsız
içe aktarılır — `streamlit` CI işi bunu hiç apt adımı olmadan kurar; tam da bu
hata bir barındırma günlüğünde değil CI'da patlasın diye.

Yine de bir sistem bağımlılığı eklerseniz, **ayrıştırıcının yorum desteklemediğini**
unutmayın: dosyayı boşluklardan böler ve her parçayı `apt-get install`'a verir;
tek bir açıklama satırı `E: Unable to locate package OpenGL,` hatasına dönüşür.
Satır başına bir çıplak paket adı, başka hiçbir şey.

### Adım adım

**1. GitHub'a koyun.**

```bash
gh repo create <hesabınız>/biosim-lab --public --source=. --remote=origin --push
```

**2. Dağıtın.** <https://share.streamlit.io> adresine gidip *New app*:

| Alan | Değer |
|---|---|
| Repository | `<hesabınız>/biosim-lab` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python version | 3.11 ya da 3.12 |

İlk derleme bağımlılıklar kurulurken 5–10 dakika sürer; sonraki `main`
itmeleri kendiliğinden yeniden dağıtılır.

**3. Denetleyin.** Dağıtılan uygulamada *Environment* sayfasını açın. Şunu
söylemelidir:

```
Instruments found via direct import (4 built-in instrument(s));
the package is not pip-installed, so third-party plugins will not be discovered
```

Bu **beklenen ve doğrudur**: Streamlit Cloud `requirements.txt`'i kurar ama
projeyi `pip install` etmez, dolayısıyla entry point meta verisi yoktur ve
`biosim_lab.registry` yerleşik cihazları doğrudan içe aktarmaya düşer.

### Kurulamayacak olanlar

Üç bileşen bilerek dışarıdadır ve `requirements.txt`'e eklemek işe yaramaz:

| Eksik | Eklemek neden işe yaramaz |
|---|---|
| **Napari** | Qt ve bir ekran sunucusu ister. Kurulur, kendini var gösterir ve yine de bir görüntüleyici açamaz. Yerelde kullanın. |
| **OpenFOAM, Elmer** | Python paketi değil, harici ikili dosyalar. Yönetilen bir sunucunun taşımadığı apt depolarından gelirler. Yoklukları akustik akışın modellenmemesi ve piezoelektrik problemin belgeli voltaj kalibrasyonuyla değiştirilmesi demektir. |
| **Cellpose, StarDist** | Kurulabilirler ama derin öğrenme çalışma zamanı çekerler. Varsayılan PyTorch tekerleği ~2.5 GB CUDA taşır ve ücretsiz katmana sığmaz. |

CPU'da Cellpose için `requirements.txt` sonundaki dört satırı açın.

### Ücretsiz katmanın içinde kalmak

- **Her simülasyon parametrelerine göre önbelleklenir**, önceki bir kaydırıcı
  konumuna dönmek anlıktır.
- **Kaydırıcılar sınırlıdır**: popülasyon başına 600 hücre, ağ çözünürlüğü 64,
  30 takip karesi, 512 piksel görüntü.
- **PyVista en ağır içe aktarımdır.** Kurulu ve ekransız çalışıyor, ama VTK'yı
  içe aktarmak birkaç yüz megabayta mal olur. Uygulama bellek yüzünden
  öldürülüyorsa `requirements.txt`'ten `pyvista`'yı çıkarmak en büyük kazançtır;
  mevcut sayfaların hiçbiri ona bağlı değildir.

### Bilinmeye değer bir tuzak

Streamlit betiğinizi **bir çalışan iş parçacığında** koşturur, ana iş parçacığında
değil. Gmsh `initialize()` çağrısında bir SIGINT işleyicisi kurar ve Python buna
yalnızca ana iş parçacığından izin verir; dolayısıyla saf bir `gmsh.initialize()`
Streamlit içinde `signal only works in main thread of the main interpreter`
hatasıyla düşer ve Gmsh doğru kurulmuş olmasına rağmen bozuk görünür.

`biosim_lab.core.geometry` bunu ele alır: iş parçacığını saptar ve ana iş
parçacığında değilse `interruptible=False` geçer. Kaybedilen tek şey ağ üretimi
sırasında Ctrl-C'dir ki bir web uygulamasında anlamsızdır. Aynı düzeltme Gmsh'ı
Jupyter çekirdeklerinde, Dask çalışanlarında ve istek işleyicilerinde de
çalışır kılar.

---

## 10. Docker

```bash
docker compose up biosim              # Panel panosu http://localhost:5006
docker compose run --rm tests         # test paketi
docker compose run --rm examples      # tüm örnekler, assets/ üretir
docker compose run --rm cli           # biosim doctor
```

Ağır çözücüler bir profilin arkasındadır, düz bir `up` onları asla derlemez:

```bash
docker compose --profile solvers build openfoam elmer
```

Ana imaj bilerek ne OpenFOAM ne Elmer içerir. `linux/arm64`'te Gmsh olmadan da
derlenir (orada wheel yoktur) ve 230 testin tamamı konteynerde geçer.

---

## 11. Sonuçları dürüstçe okumak

Bu bölüm, yanlış bir şey yayımlamanızı engelleyecek olandır.

### Uyarılar ürünün kendisidir

Turuncu `RegimeWarning` şeritleri, modelin arkasındaki bir varsayımın
zorlandığını söyler. Gürültü değildirler:

| Uyarı | Anlamı | Yapılacak |
|---|---|---|
| `k*a > 0.1` | hücreler ses dalga boyuna göre küçük değil; kuvvet fazla tahmin ediliyor | kuvvetleri mutlak değil gösterge olarak alın |
| `N pressure nodes` | kanalda birden çok toplama çizgisi var | frekansı ya da genişliği değiştirin, ya da çok çıkışlı tasarımı kabul edin |
| `Reynolds number > 1` | sürünen akış modeli artık geçerli değil | debiyi düşürün ya da OpenFOAM arka ucunu kullanın |
| `Stokes number > 0.1` | parçacık ataleti önemli | `integration` değerini `inertial` yapın |

### `analytic` iyimser moddur

Kapalı form model, çip yüzeyindeki ses şiddetini *her* yükseklikteki hücreye
uygular. Gerçek alan yukarı doğru zayıflar — 50 µm kanalda ortalaması yüzey
değerinin yalnızca **%53**'üdür.

| | geri kazanım | saflık |
|---|---|---|
| `mode: analytic` | %100 | %94.3 |
| `mode: fem` | %45.5 | %98.9 |

İkisi de hata değildir. Tasarım uzayını hızla keşfetmek için `analytic`, bir
sayıya inanmadan önce doğrulamak için `fem` kullanın.

### İki farklı belirsizlik vardır

Tek bir koşu "geri kazanım = %100" der. Arkasında iki bağımsız belirsizlik
saklanır:

- **Sayım hatası** — 300 hücrede ölçülen bir oran popülasyonun oranı değildir.
  Tohum sabitken bile vardır. Her koşunun metriklerinde `efficiency_ci_low/high`
  olarak bulunur.
- **Örnekler arası hata** — her koşu taze yarıçaplar, konumlar ve canlılık
  sonuçları çeker. `biosim run -n 5` ya da *Belirsizlik* sekmesiyle ölçülür.

> Yazılım Wilson skor aralığını kullanır, alışılmış `p ± z√(p(1−p)/n)` ifadesini
> değil. O ifade %100'de **tam olarak sıfır** genişlik verir — çalışan bir
> ayırıcının sürekli ulaştığı durum. 300/300 için Wilson `[%98.7, %100]` verir.

### `enable_vertical_arf` hücrelerinizi mahsur bırakır

Varsayılanı `false`'tur. FEM alanının gerçek bir dikey kuvveti vardır, ama bu
iki boyutlu kesit modelinde onu dengeleyecek bir kaldırma kuvveti yoktur, bu
yüzden hücreler eksenel akışın sıfır olduğu bir duvara yığılır ve çıkışa hiç
ulaşmazlar.

Açarsanız metriklerdeki **`all_cells_exited` değerini denetleyin.** `False` ise
geri kazanım ve saflık sayıları anlamsızdır.

### Canlılık da varsayımsız değildir

*Hücre güvenliği* sekmesi üç yayımlanmış hasar göstergesini hesaplar. Referans
noktasında üç marj da rahattır ve baskın neden maruziyet süresidir — hücreler
alanda 0,36 saniye kalır. **Aynı şiddette dakikalarca tutan bir tuzak bambaşka
bir meseledir.**

Modellenmeyen: liziz eşiğinin altındaki zar gözeneklenmesi (canlılık okumasını
bozar) ve ölen bir hücrenin akustik özelliklerinin değişmesi.

### Bir sayıyı aktarmadan önce kaynağını denetleyin

```bash
biosim materials            # 38 varsayım, gerekçeleriyle
```

En büyüğü: buradaki hiçbir şey sürüş voltajından akustik basıncı öngörmez.
Doğrusal bir kalibrasyon (15 Vpp → 0.45 MPa) yerine geçer ve *her* akustik
kuvveti ölçekler. Çipiniz için ölçüp `pressure_amplitude` verin.

### İki çıkışlı ayırma: büyük hücreler bir tarafa, geri kalanı diğerine

Varsayılan çip **üç** çıkışlıdır ve büyük hücreleri ortadan, basınç düğümünden
toplar. Gerçek cihazların çoğu ise **iki** çıkışlıdır: numune tek bir duvar
boyunca girer, büyük hücreler kanalı geçip düğüme doğru göç eder, küçükler etmez
ve tek bir ayırıcı iki akımı böler. Ayarlayın:

```yaml
params:
  inlet: side            # numunenin tamamı tek duvara yaslanır
  inlet_side: left
  outlet_layout: lateral_split
  split_position: 0.45   # ayırıcı, kanal genişliğinin kesri olarak
  collect_side: right    # hangi taraf toplama çıkışı
```

ya da web arayüzünde *Outlet layout → lateral_split* seçin; toplama bandı
genişliği yerine bir ayırıcı kaydırıcısı belirir.

**Ayırıcı düğümün üzerine konmaz.** Hücreler düğüme asimptotik olarak yaklaşır ve
birkaç mikron berisinde durur; bu yüzden tam düğüme konan bir ayırıcı hiçbir şey
toplamaz — model, çıplak bir %0 geri kazanım raporlamak yerine uyarır. Ayırıcıyı
iki popülasyonun indiği konumların arasına koyun; `examples/09_two_outlet_split.py`
bunu sizin için tarar:

| ayırıcı (µm) | geri kazanım | saflık |
|---|---|---|
| 60 | %100.0 | %73.9 |
| 90 | %100.0 | %92.3 |
| 120 | %98.0 | %98.7 |
| 135 | %92.3 | %99.6 |
| 150 (düğümün üzerinde) | %0.0 | — |

Geri kazanım–saflık ödünleşimi böylece açık hale gelir: ayırıcı düğmedir ve
ikisini birden en yükseğe çıkaran bir ayar yoktur. Bu geometride MCF-7'ler
144 ± 8 µm'de, alyuvarlar 53 ± 24 µm'de iner; yani hangisini önemsediğinize göre
90–135 µm arası her yer savunulabilir.

Bu arada tek taraflı giriş kendi başına da değerlidir: her hücreye göç etmek için
kanalın tam genişliğini verir, yani ayrım geometrinin izin verdiği kadar güçlü
olur; `sheath_sides` ise hücreleri **iki** duvardan başlatır ve kullanılabilir
mesafeyi yarıya indirir.

### Eğik açılı SSAW: farklı bir çıkış değil, farklı bir mekanizma

Yukarıdaki her şey duran dalganın kanalı **dik** kestiğini varsayar. O zaman
düğüm düzlemleri akışa paraleldir; hücre yana göç eder, bir düğüme varır ve
**durur**. Yer değiştirmesi, kanal ne kadar uzun ya da alan ne kadar güçlü olursa
olsun düğüm aralığıyla sınırlıdır.

IDT'leri eğmek (`tilt_angle_deg`, doi:10.1073/pnas.1413325111) cihazın yaptığı işi
değiştirir. Düğüm düzlemleri artık akışı keser, dolayısıyla bir düğümde tutulan
hücre akış yönünde ilerledikçe kanal boyunca sürüklenir:

```
dx/dz = -tan(theta)      yani tutulan hücre L uzunlukta  L * tan(theta)  kayar
```

Yer değiştirme doyuma ulaşmak yerine **kanal uzunluğuyla** büyür ve ayrım artık
"ne kadar hızlı göç ediyor" değil, **"bir düğüm onu tutabiliyor mu"** sorusudur.

**Tasarım sayısı, eğim sınırıdır.** Hareket eden bir düğüm düzleminde hücreyi
tutmak `u * tan(theta)` kadar yanal sürüklenme gerektirir; hücre ancak şu koşulda
tutulu kalır:

```
sin(theta) / cos^2(theta)  <=  pi * p0^2 * kappa_f * Phi * a^2 / (9 * mu * lambda * u)
```

Sağ taraf **a²** ile ölçeklenir, yani tutunmayı önce küçük hücreler kaybeder — ve
*ayrımı yapan tam da bu asimetridir*. `max_trappable_tilt` her popülasyonun
sınırını verir, `cutoff_radius` bunu ayırıcının kesme boyutuna çevirir; ikisi de
`diagnostics["tilt"]` içinde ve açı sıfır değilken web arayüzünde görünür.
6.632 MHz, 15 Vpp ve 5 µL/dk için:

| hücre | yarıçap | tutulma sınırı |
|---|---|---|
| MCF-7 | 9.00 µm | 16.5° |
| A549 | 7.75 µm | 9.8° |
| Akyuvar | 4.25 µm | 2.6° |
| Alyuvar | 2.78 µm | 2.4° |

2.4° ile 16.5° arasındaki herhangi bir açı tümör hücrelerini saptırır ve kan
hücrelerinin düz geçmesine izin verir. Birini seçin, kesme çapı ardından gelir:
10° için 13.7 µm.

**Sınırın ötesinde cihaz sessizce hiçbir şey yapmaz.** Tutulamayan hücre düğüm
düzlemleri arasından kayar, kuvvet ortalamada sıfırlanır ve hücre neredeyse hiç
sapmadan akıp gider — makul görünen çıktı üreten bir başarısızlık. Daha büyük
açının daha çok saptıracağını varsaymak yerine sınırı kontrol edin.

**Modelin reddettiği ya da uyardığı iki şey.** Eğimle birlikte `mode="fem"` hata
verir: Helmholtz alanı kanal kesitinde çözülür ve akış boyunca değişmez; oysa eğik
desen tanımı gereği akış boyunca değişir, dolayısıyla o ağ eğik etiketi altında
sessizce dik-IDT cevabı döndürürdü. Ayrıca pozitif açı −x yönüne saptırır; numune
sol duvardaysa **negatif** açı istersiniz. İşareti yanlış verirseniz her hücre
başladığı duvara bastırılır ve model bunu uyarır.

`examples/10_tilted_angle_ssaw.py` hem doyumun kayboluşunu hem de kazancı
gösterir: bir dalga boyu genişliğindeki 600 µm'lik kanalda dik cihaz hiçbir
ayırıcıyla %90 geri kazanıma ulaşamazken −10° %100 geri kazanım ve %100 saflık
verir.

### 38 varsayımdan hangileri gerçekten önemli

Malzeme kaynakları sayfası, DOI'ye dayandırılamayan her sayıyı listeler. Bu
dürüst bir açıklamadır ama eyleme dönük değildir: neyin bilinmediğini söyler,
bu bilgisizliğin neye mal olduğunu değil. Sıralamak için:

```bash
python examples/07_assumption_sensitivity.py
```

Kaynaksız her değeri ±%5 oynatır ve **normalize edilmiş esnekliği**
`(dY/Y)/(dX/X)` raporlar. Boyutsuz olduğu için bir yoğunlukla bir viskozite tek
eksende karşılaştırılabilir. Esneklik 1 ise girdideki %10 hata cevapta %10 hata
verir; 0 ise o girdi önemsizdir.

Çalıştırmadan önce bilinmeye değer iki sonuç:

* **CTC/RBC ayrımını tek bir varsayım oynatıyor**: MCF-7 hücre yoğunluğu
  (esneklik 0.94). Hücre yarıçapı dağılımı −0.16 ile uzak ikinci. Geri kalanı ya
  akustik model tarafından hiç kullanılmıyor ya da ölçümün çözünürlüğünün
  altında. Yani "önce neyi ölçmeliyim?" sorusunun cevabı otuz sekiz değil, tek
  bir madde: hücre hattınızı yoğunluk gradyanında bantlayın.
* **Sıralama modelin değil, çalışma noktasının bir özelliğidir.** Tasarım
  noktasında ayırıcı hedef hücrelerin neredeyse tamamını topluyor; bu tavana
  dayanmak onu her şeye karşı duyarsız kılıyor (en büyük esneklik 0.007). Marjinal
  bir noktada aynı sayı **130 kat daha fazla** önem kazanıyor. Ölçüldüğü çalışma
  noktası belirtilmeden verilen bir duyarlılık anlamsızdır — taramayı gerçekte
  çalıştığınız noktada yapın.

Tarama, kolayca yanlış yapılan iki konuda dikkatli davranır. Oynatılmış koşular
temel koşunun rastgele tohumunu yeniden kullanır; böylece fark, yeniden
örnekleme gürültüsünü değil parametrenin etkisini yalıtır. Anlamlılık ise ham
metriğin yayılımına değil, tohumlar arasındaki **eşleşmiş farka** bakılarak
kararlaştırılır; ham yayılım büyük bir çarpanla yanlış ölçüdür ve gerçek etkileri
eler.

### Cihazları zincirlemek ve hata bütçesi

Ayırma, sayma ve takip burada ayrı cihazlar; ama gerçek bir deney bunları arka
arkaya çalıştırır. Yalnızca zincirlendiklerinde görünür olan iki şey var:

```bash
python examples/08_pipeline_sort_count_track.py
```

**Ayırıcı popülasyonun yalnızca sayısını değil, kendisini değiştirir.** Radyasyon
kuvveti hücre hacmiyle, sürüklenme ise yarıçapla ölçeklenir; dolayısıyla göç hızı
`r²` ile gider ve toplama boyut seçicidir. Örnekte yüklenen süspansiyonun ortalama
çapı 11.7 µm ve CV'si 0.54 iken sayıcıya ulaşan 18.3 µm ve CV 0.12 —
**ortalamada +%56 ve 4.5 kat daha dar**. Yüklenen dağılıma göre ayarlanmış bir
sayıcı yanlış popülasyonu ölçüyor olurdu. Bu yüzden `CountStage` boyut dağılımını
bir varsayılandan değil, kendisine verilen örnekten alır.

**Belirsizlikler birleşir ve bir aşama baskın çıkar.** Her aşama farklı türde hata
katar: ayırma *binom* (sonlu sayıda hücre çıkışa ya ulaşır ya ulaşmaz), sayma
*Poisson* (görüş alanındaki `1/√N`), takip *örnekten örneğe*. Bağımsız oldukları
için karelerin toplamı olarak birleşirler — %3 ile %4, %7 değil %5 eder — ve
toplam genellikle tek bir aşamanın hâkimiyetindedir. Özet o aşamayı adıyla
söyler. Eyleme dönük kısım budur: ayırmayla sınırlı bir ölçümü daha fazla görüş
alanı görüntüleyerek kurtaramazsınız.

---

## 12. Sorun giderme

**`biosim: command not found`**
Sanal ortam etkin değil. `source .venv/bin/activate`, ya da
`python -m biosim_lab.cli` olarak çağırın.

**`0 instruments discovered`**
Paket kurulmamış. Ya `pip install -e .`, ya da yerleşikleri doğrudan içe aktaran
`biosim_lab.registry.installed_instruments()` kullanın.

**`RuntimeError: a PDMS wall layer requires gmsh`**
`pip install biosim-lab[mesh]`. Gmsh olmadan yalnızca duvarsız düz kanal
kullanılabilir — gönderilen her yapılandırmanın kullandığı da odur.

**`no usable NetCDF back-end`**
`pip install netCDF4` (ya da `h5netcdf` **ve** `h5py` — `h5netcdf` `h5py`
olmadan içe aktarılır ama yazma anında düşer).

**Simülasyon çok yavaş**
Neredeyse her zaman yüksek `fem_resolution` ile `mode: fem`. 32'den başlayın.
*Analytic* mod yavaşsa `all_cells_exited` değerine bakın: duvara sıkışmış
hücreler integrasyon penceresini muazzam uzatır.

**`DimensionalityError`**
YAML'daki bir birimin boyutu yanlış — örneğin `frequency: 300 um`. Bu, birim
denetleyicisinin işini yapmasıdır.

**Bir sıcaklık ayarladıktan sonra sonuçlar değişti**
Değişmeliydi. Viskozite 25 °C ile 37 °C arasında %22 düşer ve göç hızı onunla
ters orantılıdır. Öncesinde her şey örtük olarak 25 °C'deydi.

**Tüm hücreler tek yere gidiyor, ayrım yok**
Tanı satırına bakın. Birden çok basınç düğümü varsa kanal iki çıkışa değil
şeritlere ayırır. `f = c_SAW / (2 × genişlik)` kullanın.

**Dağıtım `E: Unable to locate package <bir İngilizce sözcük>` ile düşüyor**
`packages.txt` içinde yorum var. Ayrıştırıcının yorum desteği yoktur ve
düzyazınızı paket adı olarak okuyor. Satır başına bir çıplak paket adına indirin.

**Dağıtım `libGL.so.1: cannot open shared object file` ile düşüyor**
Tersi sorun: `packages.txt` yok ya da `libgl1` listelenmemiş.

**Bir şey değiştirdikten sonra testler düşüyor**
*Hangi* testin düştüğünü okuyun. Ezberlenmiş sayılar yerine ilişkileri
denetlerler (kuvvet ∝ r³, hücreler için Φ > 0, hiçbir hücre kaybolmaz), bu yüzden
bir düşüş genellikle bozduğunuz fiziksel özelliği adıyla söyler.

---

## 13. Genişletmek

Bir cihaz eklemek on adımdır ve **çekirdekte hiçbir değişiklik gerektirmez** —
tam anlatım [`CONTRIBUTING.md`](../CONTRIBUTING.md) içindedir. Ana hatlarıyla:

1. `biosim_lab/instruments/` altında (ya da kendi dağıtımınızda) bir paket açın.
2. Birim açıklamalı alanları olan bir `BaseConfigModel` alt sınıfı yazın.
3. Fiziği kendi modülüne koyun, her formüle DOI ekleyin.
4. Model menzil dışına çıktığında `RegimeWarning` verin.
5. `Instrument` arayüzünü uygulayın: `setup()` (idempotent) ve `run()`.
6. Kendi kaydetme/çizim kodunuz yerine çekirdek `io` ve `viz` katmanlarını kullanın.
7. `core.viz.dashboard.shell()` ile bir pano ekleyin.
8. Çalışan bir `example_config()` verin — testler onu çalıştırır.
9. `biosim_lab.instruments` altına bir entry point kaydedin.
10. Test yazın; opsiyonel bir arka uç kullanıyorsanız yokken atlanan biri dahil.

### Bozulmaması gereken kurallar

1. Tüm test paketi, hiçbir opsiyonel arka uç kurulu değilken geçer.
2. Her fiziksel sabit ya DOI'li ya açık `ASSUMPTION` etiketlidir.
3. Birimler sınırda doğrulanır; çekirdek düz SI `float` kullanır.
4. Cihazlar birbirini içe aktarmaz, çekirdek hiçbir cihazı içe aktarmaz.
5. Menzil dışındaki modeller uyarır; sessizce inandırıcı ve yanlış bir sayı
   döndürmezler.

---

## Daha fazla okuma

| Belge | İçerik |
|---|---|
| [`docs/explainer/biosim-lab-nasil-calisir.pdf`](explainer/) | Uzman olmayanlar için 31 sayfalık anlatım |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Katmanlar, sözleşmeler, veri akışı, sınırlamalar |
| [`docs/physics.md`](physics.md) | Her formül, kaynağı ve geçerlilik sınırı |
| [`docs/validation.md`](validation.md) | Ne, neye karşı, hangi toleransla doğrulandı |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | On adımda cihaz eklemek |
