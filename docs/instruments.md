# Cihaz referansı / Instrument reference

Her cihaz aynı `Instrument` arayüzünü uygular ve aynı `io` / `viz` çekirdeğini
kullanır. `biosim init <ad>` çalışan bir yapılandırma yazar.

---

## `saw_sorter` — Aşama 1, **tam**

Duran yüzey akustik dalgası (SSAW) ile mikroakışkan kanalda hücre ayırımı.
Ticari karşılığı: CTC zenginleştirme ve hücre yıkama için satılan akustik hücre
ayırıcılar. Cihaz kavramı: Li et al. (2015), PNAS 112:4970,
`doi:10.1073/pnas.1504484112`.

### Ana parametreler

| Parametre | Varsayılan | Not |
|---|---|---|
| `frequency` | 6.632 MHz | `c_SAW/(2W)` — 300 µm kanalda tek düğüm verir |
| `voltage_pp` | 15 V | `pressure_amplitude` verilirse yok sayılır |
| `pressure_amplitude` | — | Ölçülmüşse doğrudan verin; kalibrasyonu atlar |
| `channel_width` / `_height` / `_length` | 300 / 50 µm, 2 mm | `channel_length` = IDT açıklığı |
| `flow_rate` | 5 µL/dk | Şırınga pompasının kontrol ettiği büyüklük |
| `fluid` / `substrate` | `water` / `linbo3_128yx` | `core.materials` anahtarları |
| `inlet` | `sheath_sides` | `uniform`, `centre` de var |
| `collection_fraction` | 1/3 | Merkezi toplama çıkışının genişliği / kanal |
| `mode` | `analytic` | `fem` de var — bkz. aşağıdaki uyarı |
| `enable_gravity` / `_wall_repulsion` / `_secondary_bjerknes` | `false` | İsteğe bağlı ikincil etkiler |
| `enable_vertical_arf` | `false` | FEM'in dikey kuvveti; açarsanız `all_cells_exited`'ı kontrol edin |
| `populations` | MCF-7 + RBC | `cell_type`, `count`, `target` |

### Çıktılar

`metrics`: `efficiency_percent` (geri kazanım), `purity_percent`,
`enrichment_fold`, `channel_reynolds`, `particle_reynolds`,
`all_cells_exited`, ve `per_population` altında popülasyon başına
toplanma oranı, yer değiştirme ve çıkış konumu istatistikleri.

`fields`: yörüngeler `position(particle, time, axis)`, çıkış histogramı, FEM
modunda ayrıca popülasyon başına `F_x`, `F_y`, `gorkov_potential` ızgaraları.

`table`: hücre başına bir satır — yarıçap, giriş/çıkış konumu, kalış süresi,
hangi çıkışa gittiği.

### Bilmeniz gerekenler

* **`analytic` mod iyimserdir.** Taban düzlemi kuvvetini her yükseklikte
  uygular; gerçek alan yükseklikle zayıflar. `fem` modu gerçekçi olandır ve
  daha düşük geri kazanım verir (`examples/04`).
* **20 MHz + 300 µm tek düğümlü değildir** — üç düğüm sığar, platform uyarır.
* **`enable_vertical_arf` açıkken hücreler tavana yığılabilir**; bu 2-B kesit
  modelinde dengeleyici kaldırma kuvveti olmamasının sonucudur.

### Parametre taraması

```python
df, ds = instrument.sweep({"voltage_pp": [5.0, 15.0], "flow_rate": [8.3e-11, 3.3e-10]})
```

veya `biosim sweep config.yaml -p 'voltage_pp=5 V,15 V' -p 'flow_rate=5 uL/min,20 uL/min'`.

---

## `impedance_rtca` — Aşama 2, **minimal çalışır**

xCELLigence benzeri gerçek zamanlı hücre analizi. Giaever–Keese elektrot modeli
(`doi:10.1073/pnas.88.17.7896`).

### Ana parametreler

| Parametre | Varsayılan | Not |
|---|---|---|
| `frequency` | 10 kHz | Cell Index okuma frekansı |
| `n_wells` | 96 | 384 de desteklenir |
| `duration` / `n_timepoints` | 48 s / 97 | |
| `doubling_time` / `lag_time` | ~20 sa / 2 sa | Lojistik büyüme |
| `treatment_time` | 24 sa | İlaç ekleme anı |
| `concentrations` / `replicates` | 8 doz / 3 | |
| `true_ic50` / `hill_slope` | 1.0 / 1.3 | Sentetik plakayı üretmek için |
| `source_file` | `None` | Gerçek RTCA CSV/XLSX dışa aktarımı |

### Çıktılar

Cell Index zaman serisi (`cell_index(time, well)`), üç örtü düzeyinde
`|Z|(f)` spektrumu, kuyu başına endpoint tablosu, 4PL uydurmadan
`ic50`, `ic50_stderr`, `hill_slope`, `fit_r_squared`.

### Bilmeniz gerekenler

* **Endpoint IC50 maruziyet süresine bağlıdır.** Varsayılan senaryoda ilaç
  ~20 saatlik ikilenme süresine karşı 24 saat etki eder; popülasyon azalmış
  taşıma kapasitesine tam gevşemediği için uydurma ekilen değerin altında okur
  (0.75 × 1.0). Bu gerçek bir endpoint deneyinin davranışıdır.
* `Z_referans = 15 Ω` **ASSUMPTION**'dır ve yalnızca y eksenini ölçekler.

### Gerçek veriyle

```yaml
params:
  source_file: /path/to/RTCA_export.csv
```

Okuyucu, başlık satırının üstündeki metadata satırlarını atlar ve düzensiz
alan sayılarına dayanıklıdır.

---

## `cell_counter` — Aşama 3, **iskelet + demo**

Otomatik hücre sayıcı (Countess / Cellometer benzeri).

| Parametre | Varsayılan | Not |
|---|---|---|
| `source_image` | `None` | Yoksa sentetik görüntü üretir |
| `pixel_size` / `chamber_depth` | 0.65 µm / 100 µm | Neubauer geometrisi |
| `dilution_factor` | 2.0 | 1:1 tripan mavisi karışımı |
| `backend` | `classical` | `cellpose`, `stardist` opsiyonel |
| `min_diameter_um` / `max_diameter_um` | 5 / 40 | Boyut kapısı |

Çıktı: toplam / canlı / ölü sayı, `viability_percent`, `concentration_per_ml`,
boyut dağılımı, **Poisson sayım belirsizliği** (`1/√N`) ve sentetik veride
`detection_recall`.

Geri çağırmanın %100 olmaması beklenir: kenara değen hücreler atılır (geçerli
alanları yoktur) ve birkaç temas eden çift birleşir.

---

## `cell_tracker` — Aşama 3, **iskelet + demo**

Canlı hücre takibi (Incucyte / CellTracker benzeri). trackpy ile bağlama
(`doi:10.1006/jcis.1996.0217`).

| Parametre | Varsayılan | Not |
|---|---|---|
| `source_movie` | `None` | Yoksa sentetik zaman serisi üretir |
| `pixel_size` / `frame_interval` | 0.65 µm / 10 dk | |
| `search_range_px` | 12 | **Kritik parametre**: küçükse hızlı hücreleri kaybeder, büyükse kimlik karıştırır |
| `memory_frames` / `min_track_length` | 2 / 5 | |

Çıktı: iz sayısı, ortalama hız, yönelim persistansı (`net/yol`), topluluk MSD
ve anormal difüzyon üsteli `α`.

Soy ağacı (`btrack`) arayüzü tanımlıdır ama uygulanmamıştır; çağırırsanız neyin
eksik olduğunu söyleyen bir hata alırsınız.

---

## `openfoam` / `elmer` — Aşama 4, **yalnızca iskelet**

Arayüz sözleşmesi, C++ kuvvet modeli taslağı, `controlDict` ve `.sif`
şablonları hazırdır; köprüler yazılmamıştır ve `NotImplementedError` ile
şablonların yerini gösterirler. Aşama 1–3 bunlar olmadan eksiksiz çalışır.

Ne zaman gerekir:

* **OpenFOAM** — gerçek 3-B akustik akış (streaming), `Re > 1` rejimi, veya
  2-B kesitin temsil edemediği 3-B geometri (genişleme, çatallanma).
* **Elmer** — piezoelektrik IDT tam çözümü; Aşama 1'in en büyük varsayımı olan
  voltaj→basınç kalibrasyonunu kaldırır.
