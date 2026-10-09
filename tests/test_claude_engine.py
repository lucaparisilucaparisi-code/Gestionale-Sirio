"""Test del motore Claude Vision (sirio/engines/claude_engine.py) senza rete.

Il client Anthropic e' sostituito da un oggetto finto che registra le
richieste e restituisce risposte preconfezionate; un test usa il vero SDK con
un trasporto HTTP simulato per verificare il formato effettivo sul filo.
Nessun dato personale: solo nomi di fantasia.
"""

from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace
from typing import Any

import anthropic
import cv2
import httpx2
import numpy as np
import pytest

from sirio.config import model_info
from sirio.engines import claude_engine as ce
from sirio.engines import prompts
from sirio.engines.base import EngineError, PageInput
from sirio.models import DAY_FIELDS, DayRow, Header
from sirio.validation import validate
from sirio.vision.grid import TableGrid, template_grid

API_KEY = "sk-ant-api03-chiave-di-prova-0123456789"
FORBIDDEN_KWARGS = {"temperature", "top_p", "top_k", "thinking", "tool_choice", "tools", "stop_sequences"}
WORKED_DAYS = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27]  # febbraio 2026, lun-ven


# ==========================================================================
# Dati e client finti
# ==========================================================================

def row_json(g: int, pe: str | None = None, pu: str | None = None, ee: str | None = None, eu: str | None = None,
             ore: str | None = None, aa: bool = False, ao: bool = False, firma: bool = False,
             note: str | None = None, tr: bool = False, inc: tuple[str, ...] = (),
             ill: tuple[str, ...] = ()) -> dict:
    # come nello schema: valori testuali assenti = stringa vuota
    return {
        "giorno": g, "prog_entrata": pe or "", "prog_uscita": pu or "", "eff_entrata": ee or "",
        "eff_uscita": eu or "", "ore_dichiarate": ore or "", "assenza_alunno": aa, "assenza_operatore": ao,
        "firma": firma, "note": note or "", "trattino_effettivo": tr, "incerti": list(inc),
        "illeggibili": list(ill),
    }


def worked(g: int, **kw: Any) -> dict:
    base = {"pe": "8:00", "pu": "11:00", "ee": "8:00", "eu": "11:00", "ore": "3", "firma": True}
    base.update(kw)
    return row_json(g, **base)


def header_json(**kw: Any) -> dict:
    h = {
        "anno_scolastico": "2025/2026", "lotto": "1", "municipalita": "2", "ente": "Cooperativa di Prova",
        "istituto": "IC 1 Esempio", "operatore": "rossi mario", "alunno": "bianchi luca",
        "mese_anno": "02/2026", "ore_pei": "15", "sostituzione": "", "data_compilazione": "",
        "firma_coordinatore": True, "timbro_referente": True, "totale_mensile_dichiarato": "60",
        "incerti": [], "illeggibili": [],
    }
    h.update(kw)
    return {k: "" if v is None else v for k, v in h.items()}


def first_json(rows: dict[int, dict] | None = None, header: dict | None = None, is_form: bool = True,
               confidence: float = 0.93, notes: str | None = None) -> dict:
    """Lettura completa coerente (20 giorni da 3 ore = 60), con sostituzioni."""
    by_day = {g: worked(g) for g in WORKED_DAYS}
    by_day.update(rows or {})
    return {
        "is_foglio_firma": is_form,
        "header": header or header_json(),
        "rows": [by_day.get(g, row_json(g)) for g in range(1, 32)],
        "confidence": confidence,
        "ocr_notes": notes or "",
    }


def verify_json(rows: list[dict], totale: str | None = None, inc: bool = False, ill: bool = False,
                notes: str | None = None) -> dict:
    return {"rows": rows, "totale_mensile_dichiarato": totale or "", "totale_incerto": inc,
            "totale_illeggibile": ill, "ocr_notes": notes or ""}


def usage(inp: int = 1000, out: int = 500, created: int | None = 0, read: int | None = 0, **kw: Any) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=created,
                           cache_read_input_tokens=read, **kw)


def message(data: dict | str | None, stop: str = "end_turn", use: SimpleNamespace | None = None,
            model: str = "claude-opus-5-5", blocks: list | None = None) -> SimpleNamespace:
    if blocks is None:
        text = data if isinstance(data, str) else json.dumps(data)
        blocks = [SimpleNamespace(type="thinking", thinking="", signature="x"), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=use or usage(), model=model, stop_details=None)


class FakeStream:
    def __init__(self, msg: Any, events: list | None = None):
        self.msg = msg
        self.events = events or []

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self) -> Any:
        return self.msg


class FakeMessages:
    def __init__(self, owner: FakeClient, kind: str):
        self.owner = owner
        self.kind = kind

    def stream(self, **kwargs: Any) -> FakeStream:
        self.owner.calls.append((self.kind, kwargs))
        if not self.owner.responses:
            raise AssertionError("richiesta inattesa al client finto")
        item = self.owner.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, FakeStream) else FakeStream(item)


class FakeModels:
    def __init__(self, result: Any):
        self.result = result
        self.calls: list[str] = []

    def retrieve(self, model: str) -> Any:
        self.calls.append(model)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, responses: list | None = None, models_result: Any = None):
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict]] = []
        self.messages = FakeMessages(self, "messages")
        self.beta = SimpleNamespace(messages=FakeMessages(self, "beta"))
        self.models = FakeModels(models_result)


def make_page(w: int = 1240, h: int = 1754, detected: bool = True) -> PageInput:
    img = np.full((h, w, 3), 255, np.uint8)
    grid = template_grid(w, h)
    if detected:
        grid = TableGrid(width=w, height=h, col_x=grid.col_x, row_y=grid.row_y, header_top=grid.header_top,
                         total_row=grid.total_row, detected=True, score=0.95)
    return PageInput(image=img, grid=grid, source_file="prova.pdf", page=1)


def decode_images(content: list[dict]) -> list[np.ndarray]:
    out = []
    for block in content:
        if block["type"] != "image":
            continue
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/jpeg"
        raw = base64.standard_b64decode(block["source"]["data"])
        assert raw[:3] == b"\xff\xd8\xff"
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        out.append(img)
    return out


def assert_within_limits(images: list[np.ndarray]) -> None:
    for img in images:
        h, w = img.shape[:2]
        assert max(h, w) <= 2576, (w, h)
        assert w * h <= 3_750_000, (w, h)


def engine(responses: list, model: str = "claude-opus-5-5", verify: bool = True,
           effort: str = "high") -> tuple[ce.ClaudeEngine, FakeClient]:
    client = FakeClient(responses)
    return ce.ClaudeEngine(api_key=API_KEY, model=model, effort=effort, verify=verify, client=client), client


def http_error(cls: type, status: int, body: dict | None = None) -> Exception:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = body or {"type": "error", "error": {"type": "error", "message": f"errore {status}"}}
    resp = httpx2.Response(status, request=req, json=body)
    return cls(f"Error code: {status}", response=resp, body=body)


# ==========================================================================
# Schemi e prompt
# ==========================================================================

def _walk(node: Any, path: str = "$"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


@pytest.mark.parametrize("schema", [prompts.SCHEMA, prompts.VERIFY_SCHEMA], ids=["lettura", "verifica"])
def test_schema_strict(schema: dict) -> None:
    json.dumps(schema)  # serializzabile
    objects = 0
    for path, node in _walk(schema):
        bad = prompts.UNSUPPORTED_SCHEMA_KEYWORDS & set(node)
        # un campo in "properties" puo' chiamarsi come una parola chiave: si controllano solo i nodi schema
        if not path.endswith(".properties"):
            assert not bad, f"parole chiave non supportate in {path}: {bad}"
        if node.get("type") == "object" or "properties" in node and not path.endswith(".properties"):
            objects += 1
            assert node.get("additionalProperties") is False, path
            assert sorted(node["required"]) == sorted(node["properties"]), path
            assert len(node["required"]) == len(set(node["required"])), path
    assert objects >= 2


@pytest.mark.parametrize("schema", [prompts.SCHEMA, prompts.VERIFY_SCHEMA], ids=["lettura", "verifica"])
def test_schema_within_api_complexity_limits(schema: dict) -> None:
    # limiti dell'API per richiesta: <= 16 proprieta' con tipi unione, <= 24 facoltative
    unions = optional = 0
    for path, node in _walk(schema):
        if path.endswith(".properties"):
            for prop in node.values():
                if "anyOf" in prop or isinstance(prop.get("type"), list):
                    unions += 1
        if isinstance(node.get("properties"), dict) and not path.endswith(".properties"):
            optional += len(set(node["properties"]) - set(node.get("required", [])))
    assert unions <= 16 and optional <= 24
    assert unions == 0            # valori assenti = "" (nessun tipo unione)


def test_schema_mirrors_models() -> None:
    row_props = set(prompts.ROW_SCHEMA["properties"])
    assert row_props == set(DayRow.model_fields) - {"confidenza"}
    header_props = set(prompts.HEADER_SCHEMA["properties"])
    assert header_props == (set(Header.model_fields) - {"mese", "anno"}) | {"mese_anno"}
    assert prompts.ROW_SCHEMA["properties"]["incerti"]["items"]["enum"] == list(DAY_FIELDS)
    assert prompts.HEADER_SCHEMA["properties"]["illeggibili"]["items"]["enum"] == list(prompts.HEADER_FLAG_FIELDS)
    assert set(prompts.SCHEMA["properties"]) == {"is_foglio_firma", "header", "rows", "confidence", "ocr_notes"}
    assert prompts.VERIFY_SCHEMA["properties"]["rows"]["items"] is prompts.ROW_SCHEMA


def test_prompts_content() -> None:
    text = prompts.SYSTEM_PROMPT.lower()
    for word in ("illeggibili", "incerti", "trattino", "1,5", "104", "ponte di carnevale", "11:00", "14:00",
                 "napoli, __/__/____", "totale ore effettive mensili", "is_foglio_firma", "esattamente come scritto"):
        assert word in text, word
    for forbidden in ("ragionamento", "passo dopo passo", "step by step", "reasoning", "chain of thought",
                      "spiega il tuo", "pensa ad alta voce"):
        assert forbidden not in text
    t = prompts.verification_text([3, 5], ["intestazione", "giorno 3", "giorno 5"], include_total=False)
    assert "Immagine 2 = giorno 3" in t and "Immagine 3 = giorno 5" in t and "3, 5" in t
    assert "totale_incerto false" in t
    assert "approssimativi" in prompts.first_pass_text(["a"], grid_detected=False)
    assert "approssimativi" not in prompts.first_pass_text(["a"], grid_detected=True)


# ==========================================================================
# Disponibilita' e forma della richiesta
# ==========================================================================

def test_not_available_without_key() -> None:
    eng = ce.ClaudeEngine(api_key=None)
    assert eng.name == "claude"
    assert eng.is_available() == (False, "Inserisci la chiave API di Anthropic nelle Impostazioni.")
    with pytest.raises(EngineError, match="chiave API"):
        eng.extract(make_page(), lambda *_: None)
    assert ce.ClaudeEngine(api_key="   ").is_available()[0] is False
    ok, msg = ce.ClaudeEngine(api_key=API_KEY).is_available()
    assert ok and "Claude Opus 5.5" in msg and API_KEY not in msg


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_request_shape_with_fallbacks(model: str) -> None:
    eng, client = engine([message(first_json(), model=model)], model=model, verify=False)
    result = eng.extract(make_page(), lambda *_: None)
    assert len(client.calls) == 1
    kind, kw = client.calls[0]
    assert kind == "beta"
    assert kw["model"] == model
    assert kw["max_tokens"] == 32000
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert not FORBIDDEN_KWARGS & set(kw)
    assert kw["system"] == [{"type": "text", "text": prompts.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert kw["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": prompts.SCHEMA}}
    msgs = kw["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"          # nessun prefill dell'assistente
    content = msgs[0]["content"]
    assert [b["type"] for b in content] == ["image"] * 4 + ["text"]  # immagini prima del testo
    assert "Immagine 4" in content[-1]["text"]
    images = decode_images(content)
    assert_within_limits(images)
    assert result.engine == "claude" and result.model == model


def test_haiku_without_fallbacks() -> None:
    eng, client = engine([message(first_json(), model="claude-haiku-5-5")], model="claude-haiku-5-5", verify=False,
                         effort="medium")
    eng.extract(make_page(), lambda *_: None)
    kind, kw = client.calls[0]
    assert kind == "messages"
    assert "betas" not in kw and "fallbacks" not in kw
    assert not FORBIDDEN_KWARGS & set(kw)
    assert kw["output_config"]["effort"] == "medium"


def test_invalid_effort_falls_back_to_high() -> None:
    assert ce.ClaudeEngine(api_key=API_KEY, effort="massimo").effort == "high"
    assert ce.ClaudeEngine(api_key=API_KEY, effort="xhigh").effort == "xhigh"


@pytest.mark.parametrize("size", [(5000, 7070), (2480, 3508), (800, 1131)])
def test_images_within_limits_and_never_upscaled(size: tuple[int, int]) -> None:
    w, h = size
    eng, client = engine([message(first_json())], verify=False)
    eng.extract(make_page(w, h), lambda *_: None)
    images = decode_images(client.calls[0][1]["messages"][0]["content"])
    assert_within_limits(images)
    for img in images:
        assert img.shape[0] <= h and img.shape[1] <= w      # la prima lettura non ingrandisce mai
    # pagina intera: rapporto d'aspetto preservato
    assert abs(images[0].shape[1] / images[0].shape[0] - w / h) < 0.01


def test_fit_to_limits() -> None:
    big = np.zeros((4000, 3000, 3), np.uint8)
    out = ce.fit_to_limits(big)
    assert out.shape[0] * out.shape[1] <= 3_750_000 and max(out.shape[:2]) <= 2576
    strip = np.zeros((90, 1500, 3), np.uint8)
    up = ce.fit_to_limits(strip, max_upscale=2.0)
    assert max(up.shape[:2]) <= 2576
    assert up.shape[1] == 2576 and up.shape[0] == int(90 * 2576 / 1500)
    small = np.zeros((40, 300, 3), np.uint8)
    assert ce.fit_to_limits(small, max_upscale=2.0).shape[:2] == (80, 600)
    assert ce.fit_to_limits(small) is small


def test_grid_not_detected_uses_proportional_crops() -> None:
    eng, client = engine([message(first_json())], verify=False)
    eng.extract(make_page(detected=False), lambda *_: None)
    content = client.calls[0][1]["messages"][0]["content"]
    images = decode_images(content)
    assert len(images) == 4
    assert_within_limits(images)
    assert "approssimativi" in content[-1]["text"]


def test_grid_rescaled_to_page_size() -> None:
    page = make_page(1240, 1754)
    big = cv2.resize(page.image, (2480, 3508))
    page2 = PageInput(image=big, grid=page.grid, source_file="x.pdf", page=1)   # griglia a meta' risoluzione
    img, grid, detected = ce._page_and_grid(page2)
    assert detected and (grid.width, grid.height) == (2480, 3508)
    assert abs(grid.row_y[-1] - 2 * page.grid.row_y[-1]) <= 2


# ==========================================================================
# Normalizzazione
# ==========================================================================

def test_normalization() -> None:
    data = first_json(
        rows={
            3: row_json(3, "8:00", "11.00", "8,00", "11", "3", firma=True),
            4: row_json(4, "8", "11:00", "-", "–", "1,5", aa=True, firma=True, tr=False),
            10: row_json(10, "8:00", "11:00", "-", "-", None, ao=True, firma=True, note="  104 "),
            16: row_json(16, "8:00", "11:00", "-", "-", None, note="ponte di  carnevale", tr=True),
            18: row_json(18, "8:00", "11:00", "8:00", "25:00", "abc", firma=True),
            19: row_json(19, "8:00", "11:00", "8:00", "11:00", "3", firma=True, note="x",
                         ill=["note", "firma"], inc=["note"]),
        },
        header=header_json(operatore="  rossi   mario ", mese_anno="2/26", sostituzione="SÌ",
                           data_compilazione="5/3/26", anno_scolastico="2025/26", ore_pei="12,5",
                           totale_mensile_dichiarato="49,5"),
        confidence=85,
    )
    res = ce.parse_extraction(data)
    rows = {r.giorno: r for r in res.rows}
    assert len(res.rows) == 31 and [r.giorno for r in res.rows] == list(range(1, 32))
    r3 = rows[3]
    assert (r3.prog_entrata, r3.prog_uscita, r3.eff_entrata, r3.eff_uscita, r3.ore_dichiarate) == (
        "08:00", "11:00", "08:00", "11:00", 3.0)
    r4 = rows[4]
    assert r4.prog_entrata == "08:00" and r4.eff_entrata is None and r4.eff_uscita is None
    assert r4.trattino_effettivo is True and r4.ore_dichiarate == 1.5 and r4.assenza_alunno is True
    assert rows[10].note == "104" and rows[10].assenza_operatore and rows[10].ore_dichiarate is None
    assert rows[16].note == "PONTE DI CARNEVALE" and rows[16].firma is False
    r18 = rows[18]
    assert r18.eff_uscita == "25:00" and "eff_uscita" in r18.incerti     # come scritto: lo segnala E03
    assert r18.ore_dichiarate is None and "ore_dichiarate" in r18.illeggibili
    r19 = rows[19]
    assert r19.note is None and r19.illeggibili == ["note"]              # illeggibile => valore None
    assert "firma" in r19.incerti and r19.firma is True and "note" not in r19.incerti
    h = res.header
    assert h.operatore == "ROSSI MARIO" and h.alunno == "BIANCHI LUCA" and h.ente == "COOPERATIVA DI PROVA"
    assert (h.mese, h.anno) == (2, 2026)
    assert h.sostituzione == "SI" and h.data_compilazione == "05/03/2026" and h.anno_scolastico == "2025/2026"
    assert h.ore_pei == 12.5 and h.totale_mensile_dichiarato == 49.5
    assert res.confidence == 0.85 and res.is_foglio_firma is True
    assert res.ocr_notes is None and rows[1].note is None and rows[1].prog_entrata is None
    assert h.sostituzione == "SI"
    assert ce.normalize_header(header_json(sostituzione="", data_compilazione="__/__/____")).sostituzione is None
    assert ce.normalize_row({"incerti": ["Eff_Uscita", "boh"], "eff_uscita": "11:00"}, 2).incerti == ["eff_uscita"]


def test_normalization_rows_missing_duplicated_or_unordered() -> None:
    data = first_json()
    data["rows"] = [row_json(5, "8:00", "11:00"), row_json(2, "9:00", "12:00"), row_json(5),
                    row_json(40, "8:00", "9:00")]
    res = ce.parse_extraction(data)
    assert [r.giorno for r in res.rows] == list(range(1, 32))
    assert res.rows[4].prog_entrata == "08:00"           # il duplicato vuoto non sovrascrive
    assert res.rows[1].prog_entrata == "09:00"
    assert not any(r.has_content() for r in res.rows if r.giorno not in (2, 5))


def test_header_flags_and_unreadable_values() -> None:
    h = ce.normalize_header(header_json(mese_anno="feb ??", incerti=["mese_anno", "ore_pei"],
                                        illeggibili=["alunno", "timbro_referente"], ore_pei="quindici?"))
    assert h.mese == 2 and h.anno is None and "anno" in h.illeggibili
    assert h.alunno is None and "alunno" in h.illeggibili
    assert "timbro_referente" in h.incerti and h.timbro_referente is True
    assert h.ore_pei is None and "ore_pei" in h.illeggibili and "ore_pei" not in h.incerti
    assert "mese" in h.incerti
    assert not set(h.incerti) & set(h.illeggibili)


def test_not_foglio_firma_skips_verification() -> None:
    data = first_json(is_form=False, header=header_json(operatore=None, alunno=None, mese_anno=None))
    data["rows"] = [row_json(g) for g in range(1, 32)]
    eng, client = engine([message(data)], verify=True)
    res = eng.extract(make_page(), lambda *_: None)
    assert res.is_foglio_firma is False and len(client.calls) == 1 and len(res.rows) == 31


# ==========================================================================
# Lettura della risposta
# ==========================================================================

def test_text_block_selected_by_type() -> None:
    blocks = [SimpleNamespace(type="thinking", thinking="", signature="s"),
              SimpleNamespace(type="text", text=json.dumps({"a": 1}))]
    assert ce.message_json(SimpleNamespace(content=blocks)) == {"a": 1}


def test_midstream_fallback_split_text_and_note() -> None:
    full = json.dumps(first_json())
    blocks = [SimpleNamespace(type="text", text=full[:120]),
              SimpleNamespace(type="fallback", from_=SimpleNamespace(model="claude-opus-5-5"),
                              to=SimpleNamespace(model="claude-opus-5")),
              SimpleNamespace(type="text", text=full[120:])]
    eng, _ = engine([message(None, blocks=blocks, model="claude-opus-5")], verify=False)
    res = eng.extract(make_page(), lambda *_: None)
    assert res.header.operatore == "ROSSI MARIO"
    assert "modello di riserva claude-opus-5" in (res.ocr_notes or "")


def test_refusal_raises() -> None:
    eng, client = engine([message(None, stop="refusal", blocks=[])])
    with pytest.raises(EngineError, match="rifiutato"):
        eng.extract(make_page(), lambda *_: None)
    assert len(client.calls) == 1


def test_max_tokens_retry_then_success() -> None:
    eng, client = engine([message("{\"is_foglio", stop="max_tokens", use=usage(1000, 32000)),
                          message(first_json(), use=usage(1000, 4000))], verify=False)
    res = eng.extract(make_page(), lambda *_: None)
    assert [kw["max_tokens"] for _, kw in client.calls] == [32000, 64000]
    assert res.usage.output_tokens == 36000 and res.usage.input_tokens == 2000


def test_max_tokens_twice_raises() -> None:
    eng, client = engine([message("{", stop="max_tokens"), message("{", stop="max_tokens")])
    with pytest.raises(EngineError, match="interrotta"):
        eng.extract(make_page(), lambda *_: None)
    assert len(client.calls) == 2


def test_invalid_json_raises() -> None:
    eng, _ = engine([message("questa non e' una risposta JSON")])
    with pytest.raises(EngineError, match="non interpretabile"):
        eng.extract(make_page(), lambda *_: None)


def _req() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (lambda: http_error(anthropic.AuthenticationError, 401), "Chiave API di Anthropic non valida"),
        (lambda: http_error(anthropic.PermissionDeniedError, 403), "permessi"),
        (lambda: http_error(anthropic.NotFoundError, 404), "«claude-opus-5-5» non è disponibile"),
        (lambda: http_error(anthropic.RateLimitError, 429), "Limite di utilizzo"),
        (lambda: http_error(anthropic.BadRequestError, 400, {"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API."}}), "Credito Anthropic esaurito"),
        (lambda: http_error(anthropic.BadRequestError, 400, {"type": "error", "error": {
            "type": "invalid_request_error", "message": "messages: troppe immagini"}}),
         "Richiesta non accettata dal servizio Claude: messages: troppe immagini"),
        (lambda: http_error(anthropic.InternalServerError, 500), "sovraccarico"),
        (lambda: http_error(anthropic.APIStatusError, 529), "sovraccarico"),
        (lambda: http_error(anthropic.APIStatusError, 418), "codice 418"),
        (lambda: anthropic.APITimeoutError(request=_req()), "non ha risposto in tempo"),
        (lambda: anthropic.APIConnectionError(request=_req()), "Connessione a Internet assente"),
    ],
    ids=["401", "403", "404", "429", "credito", "400", "500", "529", "418", "timeout", "connessione"],
)
def test_error_mapping(exc: Any, expected: str) -> None:
    eng, _ = engine([exc()])
    with pytest.raises(EngineError) as info:
        eng.extract(make_page(), lambda *_: None)
    assert expected in str(info.value)
    assert API_KEY not in str(info.value)
    assert isinstance(info.value.__cause__, anthropic.AnthropicError)


def test_error_detail_redacts_keys() -> None:
    err = http_error(anthropic.BadRequestError, 400, {"type": "error", "error": {
        "type": "invalid_request_error", "message": f"chiave {API_KEY} rifiutata"}})
    assert API_KEY not in str(ce.map_api_error(err))


# ==========================================================================
# Consumi
# ==========================================================================

def test_usage_and_cost_accumulated_over_calls() -> None:
    first = first_json(rows={3: worked(3, eu="14:00")})            # E01 -> verifica
    eng, client = engine([
        message(first, use=usage(1200, 3000, created=2500, read=0)),
        message(verify_json([worked(3)]), use=usage(800, 600, created=0, read=2500)),
    ])
    res = eng.extract(make_page(), lambda *_: None)
    assert len(client.calls) == 2
    info = model_info("claude-opus-5-5")
    expected = ((1200 + 1.25 * 2500) * info["input"] + 3000 * info["output"]
                + (800 + 0.1 * 2500) * info["input"] + 600 * info["output"]) / 1e6
    assert res.usage.cost_usd == pytest.approx(expected, rel=1e-6)
    assert res.usage.input_tokens == 1200 + 2500 + 800 + 2500
    assert res.usage.output_tokens == 3600
    assert res.usage.seconds >= 0


def test_usage_none_cache_fields_and_iterations() -> None:
    meter = ce._UsageMeter("claude-opus-5-5")
    meter.add(SimpleNamespace(model="claude-opus-5-5", usage=usage(100, 10, created=None, read=None)))
    assert meter.input_tokens == 100 and meter.output_tokens == 10
    its = [SimpleNamespace(type="message", model="claude-opus-5-5", input_tokens=1000, output_tokens=0,
                           cache_creation_input_tokens=0, cache_read_input_tokens=0),
           SimpleNamespace(type="fallback_message", model="claude-haiku-5-5", input_tokens=1000, output_tokens=100,
                           cache_creation_input_tokens=None, cache_read_input_tokens=None)]
    meter2 = ce._UsageMeter("claude-opus-5-5")
    meter2.add(SimpleNamespace(model="claude-haiku-5-5", usage=usage(1000, 100, iterations=its)))
    opus, haiku = model_info("claude-opus-5-5"), model_info("claude-haiku-5-5")
    assert meter2.input_tokens == 2000
    expected = (1000 * opus["input"] + 1000 * haiku["input"] + 100 * haiku["output"]) / 1e6
    assert meter2.cost_usd == pytest.approx(expected)


# ==========================================================================
# Verifica incrociata
# ==========================================================================

def _parse(data: dict) -> tuple[Header, list[DayRow]]:
    res = ce.parse_extraction(data)
    return res.header, res.rows


def test_plan_clean_document_needs_no_verification() -> None:
    header, rows = _parse(first_json())
    anomalies, totals = validate(header, rows)
    assert totals.ore_dichiarate == 60
    assert not ce.plan_verification(header, rows).needed, [a.codice for a in anomalies]


def test_plan_selects_rows_by_code_and_flags() -> None:
    header, rows = _parse(first_json(rows={
        3: worked(3, eu="14:00"),                    # E01
        5: worked(5, inc=("firma",)),                # incerto
        9: worked(9, firma=False),                   # W01
        12: worked(12, ore=None, ill=("ore_dichiarate",)),  # illeggibile
        13: worked(13, aa=True, ao=True),            # W11 / W05
    }, header=header_json(totale_mensile_dichiarato="57")))   # somma coerente: niente E02
    plan = ce.plan_verification(header, rows)
    assert plan.days == [3, 5, 9, 12, 13]
    assert plan.include_total is False
    header2, rows2 = _parse(first_json(rows={3: worked(3, eu="14:00")},
                                       header=header_json(totale_mensile_dichiarato="60")))
    plan2 = ce.plan_verification(header2, rows2)
    assert plan2.days == [3] and plan2.include_total is False


def test_plan_e02_includes_all_days_with_hours() -> None:
    header, rows = _parse(first_json(header=header_json(totale_mensile_dichiarato="61")))
    plan = ce.plan_verification(header, rows)
    assert plan.include_total and plan.days == WORKED_DAYS


def test_verification_request_and_merge() -> None:
    page = make_page()
    first = first_json(rows={3: worked(3, eu="14:00"), 5: worked(5, inc=("firma",))},
                       header=header_json(totale_mensile_dichiarato="60"))
    reread = verify_json([worked(3), worked(5)])
    eng, client = engine([message(first), message(reread)])
    events: list[tuple[float, str]] = []
    res = eng.extract(page, lambda f, m: events.append((f, m)))

    assert len(client.calls) == 2
    kind, kw = client.calls[1]
    assert kind == "beta" and kw["fallbacks"] == "default"
    assert kw["system"] == client.calls[0][1]["system"]                 # stesso prompt di sistema (cache)
    assert kw["output_config"]["format"]["schema"] == prompts.VERIFY_SCHEMA
    content = kw["messages"][0]["content"]
    text = content[-1]["text"]
    assert "Immagine 1 = intestazione della tabella" in text
    assert "Immagine 2 = giorno 3" in text and "Immagine 3 = giorno 5" in text
    assert "Immagine 4" not in text and "Non è richiesta la rilettura del totale" in text
    images = decode_images(content)
    assert len(images) == 3
    assert_within_limits(images)
    grid = page.grid
    source_w = grid.col_x[-1] - grid.col_x[0]
    for strip in images[1:3]:
        assert strip.shape[1] > source_w                              # strisce ingrandite
        assert strip.shape[0] < strip.shape[1] / 5

    rows = {r.giorno: r for r in res.rows}
    assert rows[3].eff_uscita == "11:00" and rows[3].incerti == []      # rilettura coerente e non segnalata
    assert rows[5].incerti == [] and rows[5].firma is True              # valori concordi
    assert "uscita effettiva 14:00 → 11:00" in (res.ocr_notes or "")
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions)
    for f in (0.05, 0.15, 0.7, 0.75, 0.95, 1.0):
        assert f in fractions
    assert all(isinstance(m, str) and m for _, m in events)


def test_verification_strips_follow_the_rows() -> None:
    page = make_page()
    g = page.grid
    page.image[g.row_y[6] + 3:g.row_y[7] - 3, g.col_x[0] + 3:g.col_x[-1] - 3] = (0, 0, 255)   # giorno 7 in rosso
    crops = ce.verification_crops(page.image, g, True, [7, 8], include_total=False)
    assert [lbl for lbl, _ in crops] == ["intestazione della tabella (nomi delle colonne)", "giorno 7", "giorno 8"]

    def red(img: np.ndarray) -> float:
        return float(((img[..., 2] > 200) & (img[..., 0] < 60)).mean())

    for _, crop in crops[1:]:
        assert crop.shape[0] < crop.shape[1] / 5          # una sola riga con poco margine
    middle = [crop[crop.shape[0] // 2] for _, crop in crops]
    assert red(middle[1]) > 0.9                          # la striscia del giorno 7 e' centrata sul giorno 7
    assert red(middle[2]) == 0.0 and red(crops[2][1]) < 0.3
    assert red(crops[0][1]) == 0.0


def test_verification_disabled() -> None:
    eng, client = engine([message(first_json(rows={3: worked(3, eu="14:00")}))], verify=False)
    res = eng.extract(make_page(), lambda *_: None)
    assert len(client.calls) == 1 and res.rows[2].eff_uscita == "14:00"


def test_verification_failure_keeps_first_reading() -> None:
    eng, client = engine([message(first_json(rows={3: worked(3, eu="14:00")})),
                          http_error(anthropic.RateLimitError, 429)])
    res = eng.extract(make_page(), lambda *_: None)
    assert len(client.calls) == 2
    assert res.rows[2].eff_uscita == "14:00"
    assert "Verifica incrociata non eseguita" in (res.ocr_notes or "")


def test_verification_total_reread() -> None:
    first = first_json(header=header_json(totale_mensile_dichiarato="66"))
    reread = verify_json([worked(g) for g in WORKED_DAYS], totale="60")
    eng, client = engine([message(first), message(reread)])
    res = eng.extract(make_page(), lambda *_: None)
    text = client.calls[1][1]["messages"][0]["content"][-1]["text"]
    assert "Rileggi anche il valore" in text
    assert res.header.totale_mensile_dichiarato == 60 and "totale_mensile_dichiarato" not in res.header.incerti
    assert "totale mensile 66 → 60" in (res.ocr_notes or "")


def _rows(**kw: Any) -> tuple[DayRow, DayRow]:
    a = ce.normalize_row(worked(3, **kw.get("a", {})), 3)
    b = ce.normalize_row(worked(3, **kw.get("b", {})), 3)
    return a, b


HEADER = Header(mese=2, anno=2026, operatore="ROSSI MARIO", alunno="BIANCHI LUCA")


def test_merge_agreeing_values_drop_uncertainty() -> None:
    a, b = _rows(a={"inc": ("eff_uscita",)}, b={})
    merged, changes = ce.merge_row(HEADER, a, b)
    assert merged.incerti == [] and changes == []
    a, b = _rows(a={"inc": ("eff_uscita",)}, b={"inc": ("eff_uscita",)})
    assert ce.merge_row(HEADER, a, b)[0].incerti == []            # concordi e riga coerente


def test_merge_disagreement_rules() -> None:
    # rilettura diversa, coerente, non segnalata -> vale la rilettura, nessun dubbio
    a, b = _rows(a={"eu": "14:00", "inc": ("eff_uscita",)}, b={})
    merged, changes = ce.merge_row(HEADER, a, b)
    assert merged.eff_uscita == "11:00" and merged.incerti == [] and changes == ["uscita effettiva 14:00 → 11:00"]
    # rilettura diversa ma segnalata incerta -> resta incerto
    a, b = _rows(a={"eu": "14:00"}, b={"inc": ("eff_uscita",)})
    merged, _ = ce.merge_row(HEADER, a, b)
    assert merged.eff_uscita == "11:00" and merged.incerti == ["eff_uscita"]
    # rilettura diversa ma riga incoerente -> resta incerto
    a, b = _rows(a={"eu": "11:00"}, b={"eu": "12:00"})
    merged, _ = ce.merge_row(HEADER, a, b)
    assert merged.eff_uscita == "12:00" and merged.incerti == ["eff_uscita"]
    # una lettura vede la firma e l'altra no -> sempre incerto
    a, b = _rows(a={"firma": True}, b={"firma": False})
    merged, _ = ce.merge_row(HEADER, a, b)
    assert merged.firma is False and "firma" in merged.incerti


def test_merge_illegible_rules() -> None:
    # illeggibile in entrambe -> illeggibile, valore None
    a, b = _rows(a={"note": None, "ill": ("note",)}, b={"note": None, "ill": ("note",)})
    merged, _ = ce.merge_row(HEADER, a, b)
    assert merged.note is None and merged.illeggibili == ["note"] and merged.incerti == []
    # illeggibile solo nella rilettura -> resta il valore della prima, incerto
    a, b = _rows(a={"note": "104"}, b={"note": None, "ill": ("note",)})
    merged, _ = ce.merge_row(HEADER, a, b)
    assert merged.note == "104" and merged.incerti == ["note"] and merged.illeggibili == []
    # illeggibile solo nella prima, letto e coerente nella rilettura -> valore riletto
    a, b = _rows(a={"ore": None, "ill": ("ore_dichiarate",)}, b={})
    merged, _ = ce.merge_row(HEADER, a, b)
    assert merged.ore_dichiarate == 3.0 and merged.incerti == [] and merged.illeggibili == []
    # illeggibile nella prima e vuoto nella rilettura -> vuoto ma incerto
    a, b = _rows(a={"note": None, "ill": ("note",)}, b={"note": None})
    merged, _ = ce.merge_row(HEADER, a, b)
    assert merged.note is None and merged.incerti == ["note"] and merged.illeggibili == []


def test_empty_reread_of_filled_row_keeps_first_reading() -> None:
    header, rows = _parse(first_json(rows={3: worked(3, eu="14:00")}))
    reread = ce.parse_verification(verify_json([row_json(3)]), [3], include_total=False)
    _, merged, notes = ce.merge_verification(header, rows, reread)
    assert merged[2] == rows[2]
    assert notes == ["giorno 3: rilettura senza dati, mantenuta la prima lettura"]


def test_change_notes_mark_illegible_values() -> None:
    a, b = _rows(a={"eu": None, "ill": ("eff_uscita",)}, b={})
    merged, changes = ce.merge_row(HEADER, a, b)
    assert merged.eff_uscita == "11:00" and changes == ["uscita effettiva illeggibile → 11:00"]


def test_parse_verification_positional_and_total() -> None:
    data = verify_json([row_json(0, "8:00", "11:00"), row_json(0, "9:00", "12:00")], totale="49,5", inc=True)
    rr = ce.parse_verification(data, [4, 7], include_total=True)
    assert sorted(rr.rows) == [4, 7] and rr.rows[7].prog_entrata == "09:00"
    assert rr.totale == 49.5 and rr.totale_incerto and not rr.totale_illeggibile
    rr2 = ce.parse_verification(verify_json([], totale=None, ill=True), [4], include_total=True)
    assert rr2.totale is None and rr2.totale_illeggibile and rr2.rows == {}


# ==========================================================================
# Verifica della chiave (nessun token consumato)
# ==========================================================================

def test_api_key_check() -> None:
    ok, msg = ce.test_api_key(API_KEY, "claude-opus-5-5",
                              client=FakeClient(models_result=SimpleNamespace(display_name="Claude Opus 5.5")))
    assert ok and "Claude Opus 5.5" in msg
    client = FakeClient(models_result=http_error(anthropic.NotFoundError, 404))
    ok, msg = ce.test_api_key(API_KEY, "claude-haiku-5-5", client=client)
    assert not ok and "«claude-haiku-5-5» non è disponibile" in msg and client.models.calls == ["claude-haiku-5-5"]
    ok, msg = ce.test_api_key(API_KEY, "claude-opus-5-5",
                              client=FakeClient(models_result=http_error(anthropic.AuthenticationError, 401)))
    assert not ok and "non valida" in msg
    ok, msg = ce.test_api_key(API_KEY, "claude-opus-5-5",
                              client=FakeClient(models_result=anthropic.APIConnectionError(request=_req())))
    assert not ok and "Connessione" in msg
    assert ce.test_api_key("  ", "claude-opus-5-5")[0] is False


# ==========================================================================
# Formato sul filo con il vero SDK (trasporto HTTP simulato)
# ==========================================================================

def _sse(text: str, model: str) -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_prova", "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 3000, "output_tokens": 1, "cache_creation_input_tokens": 2000,
                      "cache_read_input_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": text[:50]}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": text[50:]}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 2500}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-haiku-5-5"])
def test_real_sdk_wire_format(model: str) -> None:
    seen: list[httpx2.Request] = []
    payload = json.dumps(first_json())

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(payload, model))

    client = anthropic.Anthropic(api_key=API_KEY, max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    eng = ce.ClaudeEngine(api_key=API_KEY, model=model, verify=False, client=client)
    res = eng.extract(make_page(), lambda *_: None)
    assert res.header.operatore == "ROSSI MARIO" and len(res.rows) == 31
    assert res.usage.input_tokens == 5000 and res.usage.output_tokens == 2500
    req = seen[0]
    body = json.loads(req.content)
    assert set(body) - {"stream"} <= {"model", "max_tokens", "system", "messages", "output_config", "fallbacks"}
    assert not FORBIDDEN_KWARGS & set(body)
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    if model == "claude-opus-5-5":
        assert req.url.params.get("beta") == "true"
        assert "server-side-fallback-2026-07-01" in req.headers.get("anthropic-beta", "")
        assert body["fallbacks"] == "default"
    else:
        assert "fallbacks" not in body and "anthropic-beta" not in req.headers
    assert req.headers.get("x-api-key") == API_KEY


# ==========================================================================
# Scansione reale (facoltativa: SIRIO_SAMPLE_PDF)
# ==========================================================================

def test_real_sample_image_preparation() -> None:
    path = os.environ.get("SIRIO_SAMPLE_PDF")
    if not path or not os.path.exists(path):
        pytest.skip("SIRIO_SAMPLE_PDF non impostata")
    from sirio.pdf_io import iter_pages  # noqa: PLC0415
    from sirio.vision.grid import detect_grid  # noqa: PLC0415
    from sirio.vision.preprocess import normalize_page  # noqa: PLC0415

    with open(path, "rb") as fh:
        _, img = next(iter_pages(fh.read(), "esempio.pdf"))
    img = normalize_page(img)
    grid = detect_grid(img)
    page = PageInput(image=img, grid=grid, source_file="esempio.pdf", page=1)
    eng, client = engine([message(first_json(rows={3: worked(3, eu="14:00")})), message(verify_json([worked(3)]))])
    eng.extract(page, lambda *_: None)
    first = decode_images(client.calls[0][1]["messages"][0]["content"])
    strips = decode_images(client.calls[1][1]["messages"][0]["content"])
    assert len(first) == 4 and len(strips) == 2
    assert_within_limits(first + strips)
    if grid.detected:
        rh = (grid.row_y[-1] - grid.row_y[0]) / 31
        # striscia del giorno 3: circa 1,7 righe di altezza, ingrandita
        scale = strips[1].shape[1] / (grid.col_x[-1] - grid.col_x[0] + 0.03 * img.shape[1])
        assert 1.2 * rh < strips[1].shape[0] / scale < 2.2 * rh
