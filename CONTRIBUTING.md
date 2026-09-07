# Contributing to biosim-lab

Katkı için teşekkürler. / Thanks for contributing.

Kod ve docstring'ler **İngilizce**; kullanıcı belgeleri Türkçe + İngilizce.
Code and docstrings are in **English**; user-facing docs are bilingual.

---

## Yeni bir cihaz eklemek — 10 adım / Adding a new instrument in 10 steps

Çekirdekte **hiçbir değişiklik gerekmez**. Örnek olarak bir "flow cytometer"
eklediğimizi varsayalım. The core needs **no changes at all**.

### 1. Paket iskeletini oluşturun

```
biosim_lab/instruments/flow_cytometer/
    __init__.py
    instrument.py
    physics.py        # veya physics/ alt paketi
    dashboard.py
```

Üçüncü taraf bir pakette de yapabilirsiniz — `biosim_lab` içinde olması şart
değil, yalnızca entry point yeterlidir.

### 2. Yapılandırma modelini yazın

`BaseConfigModel`'den türeyin ve birim taşıyan alanlar için hazır takma adları
kullanın; böylece `"20 MHz"` gibi dizeler `pint` ile doğrulanır ve çekirdeğe SI
`float` olarak geçer.

```python
from biosim_lab.core.config import BaseConfigModel, Frequency, Length

class FlowCytometerParams(BaseConfigModel):
    excitation_wavelength: Length = 488e-9
    sheath_flow: VolumeFlow = 1e-8
    n_events: int = 10_000
```

`extra="forbid"` mirastan gelir: yapılandırmadaki bir yazım hatası sessizce
yutulmaz, hata verir.

### 3. Fiziği ayrı bir modüle yazın, her formüle DOI koyun

Fizik çekirdeği **birimsiz `float`/`ndarray` alır ve SI döndürür** (birim
denetimi sınırda yapılır — ARCHITECTURE.md §8). Her fonksiyonun docstring'inde
kaynak DOI'si bulunmalıdır:

```python
def scattering_cross_section(radius: float, wavelength: float, m: complex) -> float:
    """Mie scattering cross-section [m^2].

    Reference: Bohren & Huffman, *Absorption and Scattering of Light by Small
    Particles*, doi:10.1002/9783527618156, ch. 4.
    """
```

Kaynağı olmayan bir sayı kullanacaksanız `materials.py`'deki gibi
**`Provenance(assumption=...)`** ile etiketleyin ve *neden* varsaydığınızı
yazın. `tests/test_materials.py` DOI'siz ve etiketsiz değeri reddeder.

### 4. Geçerlilik sınırlarını denetleyin

Modelinizin geçerli olmadığı bir rejimde sessizce yanlış sayı üretmeyin;
`RegimeWarning` verin ve ne yapılması gerektiğini söyleyin:

```python
from biosim_lab.core.plugin import RegimeWarning

if size_parameter > 0.1:
    warnings.warn(
        f"k*a = {size_parameter:.3g} > 0.1: the Rayleigh approximation is being "
        "stretched; use the full Mie solution.",
        RegimeWarning, stacklevel=2,
    )
```

### 5. `Instrument` arayüzünü uygulayın

```python
from biosim_lab.core.plugin import Instrument, InstrumentResult

class FlowCytometer(Instrument):
    name = "flow_cytometer"              # entry point adıyla aynı olmalı
    display_name = "Flow cytometer"
    description = "..."
    ConfigModel = FlowCytometerParams

    def setup(self) -> None:
        if self._is_set_up:
            return                        # setup() idempotent olmalı
        self.params = FlowCytometerParams.model_validate(self.config.params)
        self._is_set_up = True

    def run(self) -> InstrumentResult:
        self._ensure_setup()
        ...
        self._result = InstrumentResult(fields=..., metrics=..., table=..., meta=...)
        return self._result
```

`fields` bir `xarray.Dataset` (her değişkende `units` özniteliği), `metrics`
skaler bir sözlük, `table` nesne başına bir satırlık `DataFrame`'dir. Böylece
`io.save_result()` hiçbir değişiklik olmadan çalışır.

### 6. Çekirdek `io` ve `viz` katmanını kullanın

Kendi kaydetme veya çizim kodunuzu yazmayın:

```python
from biosim_lab.core.io import save_result, read_image_stack
from biosim_lab.core.viz.curves import timeseries_figure, well_plate_heatmap
from biosim_lab.core.viz.theme import color_for
```

Renk politikası bağlayıcıdır: kategorik renk **varlığa** bağlanır, sıralamasına
değil (`color_for()` bunu sağlar); büyüklük için tek renkli sıralı skala,
işaretli veri için nötr orta noktalı ıraksak skala; iki farklı ölçekli büyüklük
asla tek grafikte iki y ekseni ile gösterilmez — iki grafik yapın.

### 7. Panoyu ekleyin

`core.viz.dashboard.shell()` ortak kabuğu verir: başlık, kontrol satırı, metrik
kutuları, sekmeli grafik alanı, tablo.

```python
def dashboard(self) -> Any:
    from biosim_lab.instruments.flow_cytometer.dashboard import build_dashboard
    return build_dashboard(self)
```

Panel'i modül seviyesinde `import` etmeyin — `require_panel()` içinden yapın ki
çekirdek Panel olmadan da yüklenebilsin.

### 8. `example_config()` yazın

`biosim init flow_cytometer` bunu yazar ve testler **çalıştırır**; dolayısıyla
gerçekten çalışan, anlamlı bir yapılandırma olmalıdır.

```python
@classmethod
def example_config(cls) -> dict[str, Any]:
    return {"name": "fc_demo", "instrument": cls.name, "params": {...}}
```

### 9. Entry point'i kaydedin

```toml
[project.entry-points."biosim_lab.instruments"]
flow_cytometer = "biosim_lab.instruments.flow_cytometer:FlowCytometer"
```

`pip install -e .` sonrası `biosim list` ve `biosim doctor` cihazı görür.
Ayrı bir pakette geliştiriyorsanız aynı grup adını kullanmanız yeterlidir.

### 10. Test yazın

En az şunlar:

* formüllerin **özdeşlikleri** (ezberlenmiş sayı değil): limitler, ölçekleme,
  işaret, korunum;
* uçtan uca bir koşu: `example_config()` → `run()` → beklenen metrikler;
* geçersiz yapılandırmanın reddedildiği;
* opsiyonel bir arka uç kullanıyorsanız, **kurulu değilken** testin atlandığı.

```bash
pytest tests/test_flow_cytometer.py
```

---

## Yeni bir çözücü arka ucu eklemek

`Solver` sözleşmesini uygulayın (`core/solver.py`), `required_executables` /
`required_modules` bildirin ve `biosim_lab.solvers` grubuna kaydedin. Kurulu
değilken `get_solver()` bir `UnavailableSolver` döndürür ve çağıran taraf
analitik yola düşer — **`ImportError` fırlatmayın**.

```python
class MySolver(Solver):
    name = "mysolver"
    kind = "flow"
    required_executables = ("mysolver-bin",)

    def setup(self, mesh, bc): ...
    def run(self): ...
    def fields(self) -> xr.Dataset: ...
```

---

## Değişmez kurallar / Invariants

Bunlar bozulursa PR birleştirilmez:

1. **Ağır çözücüler çekirdeğe sızmaz.** Tüm test paketi OpenFOAM, Elmer,
   Napari, Cellpose ve StarDist kurulu **değilken** geçmelidir.
2. **Kaynaksız sayı yok.** Her fiziksel sabit ya DOI'li ya `ASSUMPTION`
   etiketli.
3. **Birimler sınırda.** Genel API `pint` ile denetler, çekirdek SI `float`
   kullanır. İkinci bir `UnitRegistry` oluşturmayın — `core.units.ureg` tektir.
4. **Cihazlar birbirini import etmez** ve çekirdek hiçbir cihazı import etmez.
   Tek yön: instruments → core → solvers.
5. **Sessiz yanlış sonuç yok.** Model geçerlilik dışına çıkarsa `RegimeWarning`.
6. `setup()` idempotent, `run()` yeniden çalıştırılabilir olmalı.

## Kod stili

```bash
ruff check biosim_lab tests
ruff format biosim_lab tests
mypy biosim_lab            # tavsiye edilir, zorunlu değil
```

Satır uzunluğu 96. Fizik gösterimindeki tek harfli adlar (`N`, `E`, `k`, `I`)
kasıtlı olarak serbesttir (`pyproject.toml`'da `ruff.lint.ignore`).

## Commit ve PR

* Bir PR bir şey yapsın; fizik değişikliği ile biçim değişikliğini karıştırmayın.
* Sayısal bir sonucu değiştiren PR, **hangi testin bu değişikliği yakaladığını**
  ve neden yeni değerin doğru olduğunu açıklasın.
* Yeni bir varsayım ekliyorsanız `biosim materials` çıktısında görünmelidir.
