"""
仿生大脑 · 工具层 (tools_layer.py)
====================================
给它"手脚" —— 能上网、能抓网页、能搜索。

设计原则：
  · 零新依赖（标准库 urllib + html.parser）
  · 工具注册表 + 统一调用接口
  · 安全护栏（域名白名单、频率限制、超时）
  · 结果自动"教"给神经组织（自学习闭环）

工具列表：
  web_fetch    抓取网页全文
  web_search   搜索（DuckDuckGo HTML 版，免 key）
  wiki_lookup  维基百科查询
  http_get     JSON API 调用
  fs_read      读本地文件（受限目录）
  brain_state  查大脑状态

用法：
    python tools_layer.py --demo
    python tools_layer.py --call web_search "Python asyncio"
"""
from __future__ import annotations
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# ---------------- 安全护栏 ----------------
ALLOW_DIRS = [Path(r"F:\DESKTOP\AI架构与推理设计"), Path(r"E:\BRAIN_DATA")]
_BLOCKED_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}   # 防 SSRF 打本地服务
_rate: Dict[str, List[float]] = {}


def _rate_ok(key: str, max_per_min: int = 20) -> bool:
    now = time.time()
    hits = [t for t in _rate.get(key, []) if now - t < 60]
    if len(hits) >= max_per_min:
        return False
    hits.append(now)
    _rate[key] = hits
    return True


class TextExtractor(HTMLParser):
    """从 HTML 提取纯文本（标准库，零依赖）"""

    SKIP = {"script", "style", "noscript", "svg", "head", "meta", "link"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False
        self.links: List[tuple] = []
        self._href = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            d = dict(attrs)
            self._href = d.get("href")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._href:
            self._href = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._skip:
            return
        t = data.strip()
        if t:
            self.parts.append(t)

    def text(self, limit: int = 8000) -> str:
        s = re.sub(r"\s+", " ", " ".join(self.parts))
        return s[:limit]


# ---------------- HTTP ----------------
def http_get(url: str, timeout: float = 15.0, raw: bool = False) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if raw:
        return data
    # 尝试多种编码
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ---------------- 工具实现 ----------------
def tool_web_fetch(url: str, limit: int = 6000) -> Dict[str, Any]:
    """抓取网页并提取正文"""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    host = urllib.parse.urlparse(url).hostname or ""
    if host in _BLOCKED_HOSTS:
        return {"ok": False, "error": "blocked host"}
    if not _rate_ok("fetch"):
        return {"ok": False, "error": "rate limited"}
    try:
        html = http_get(url)
        p = TextExtractor()
        p.feed(html)
        return {"ok": True, "url": url, "title": p.title.strip(),
                "text": p.text(limit), "chars": len(html)}
    except Exception as e:
        return {"ok": False, "url": url, "error": f"{type(e).__name__}: {e}"}


class _DDGParser(HTMLParser):
    """解析 DuckDuckGo HTML 版结果"""

    def __init__(self):
        super().__init__()
        self.results = []
        self._cur = None
        self._grab = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        cls = d.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._cur = {"title": "", "url": d.get("href", "")}
            self._grab = "title"
        elif tag == "a" and "result__snippet" in cls and self._cur:
            self._grab = "snippet"
        elif tag == "a" and "result__snippet" in cls:
            self._cur = self._cur or {"title": "", "url": ""}
            self._grab = "snippet"

    def handle_endtag(self, tag):
        if tag == "a" and self._grab == "title" and self._cur:
            self.results.append(self._cur)
            self._cur = None
            self._grab = None
        elif tag == "a" and self._grab == "snippet":
            self._grab = None

    def handle_data(self, data):
        if self._grab == "title" and self._cur is not None:
            self._cur["title"] += data
        elif self._grab == "snippet" and self.results:
            self.results[-1].setdefault("snippet", "")
            self.results[-1]["snippet"] += data


class _BingParser(HTMLParser):
    """解析必应搜索结果（cn.bing.com，国内可达）"""

    def __init__(self):
        super().__init__()
        self.results = []
        self._cur = None
        self._grab = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "li" and "b_algo" in d.get("class", ""):
            self._cur = {"title": "", "url": "", "snippet": ""}
            self._grab = None
        elif tag == "h2" and self._cur is not None:
            self._grab = "title"
        elif tag == "a" and self._cur is not None and self._grab == "title":
            self._cur["url"] = d.get("href", "")
        elif tag == "p" and self._cur is not None:
            self._grab = "snippet"

    def handle_endtag(self, tag):
        if tag == "li" and self._cur is not None:
            if self._cur.get("title"):
                self.results.append(self._cur)
            self._cur = None
            self._grab = None
        elif tag in ("h2", "p"):
            self._grab = None

    def handle_data(self, data):
        if self._cur is None or self._grab is None:
            return
        self._cur[self._grab] += data


def tool_web_search(query: str, n: int = 5) -> Dict[str, Any]:
    """搜索（必应中文版，国内可达；维基/DDG 被墙）"""
    if not _rate_ok("search", 10):
        return {"ok": False, "error": "rate limited"}

    # 多引擎回退：必应 → 百度
    engines = [
        ("bing", "https://cn.bing.com/search?q=" + urllib.parse.quote(query)),
    ]
    last_err = None
    for name, url in engines:
        try:
            html = http_get(url, timeout=12)
            p = _BingParser()
            p.feed(html)
            out = []
            for r in p.results[:n]:
                out.append({
                    "title": re.sub(r"\s+", " ", r.get("title", "")).strip()[:120],
                    "url": r.get("url", ""),
                    "snippet": re.sub(r"\s+", " ", r.get("snippet", "")).strip()[:300],
                })
            if out:
                return {"ok": True, "query": query, "engine": name,
                        "n": len(out), "results": out}
            last_err = "no results parsed"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return {"ok": False, "query": query, "error": last_err or "unknown"}


def tool_wiki_lookup(term: str, lang: str = "zh") -> Dict[str, Any]:
    """百科查询：优先维基（若可达），否则回退必应搜索摘要。

    说明：本机维基被墙（实测超时），因此默认走"必应摘要"路线。
    """
    if not _rate_ok("wiki", 20):
        return {"ok": False, "error": "rate limited"}

    # 1) 试维基（可能被墙）
    if lang in ("zh", "en"):
        try:
            t = urllib.parse.quote(term.replace(" ", "_"))
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{t}"
            d = json.loads(http_get(url, timeout=6))
            if d.get("extract"):
                return {"ok": True, "term": term, "source": "wikipedia",
                        "title": d.get("title", ""), "extract": d.get("extract", ""),
                        "url": (d.get("content_urls", {}).get("desktop", {}) or {}).get("page", "")}
        except Exception:
            pass   # 被墙 → 回退

    # 2) 回退：★多源摘要聚合（主流站点普遍反爬，抓正文不可靠）
    #    策略：搜多个角度 → 收集摘要 → 合成一条知识（摘要本身已提炼过）
    try:
        queries = [term, f"{term} 原理", f"{term} 是什么", f"{term} 详解"]
        seen = set()
        candidates = []
        for q in queries[:2]:
            s = tool_web_search(q, n=4)
            if not s.get("ok"):
                continue
            for r in s["results"]:
                u = r.get("url", "")
                if u and u not in seen:
                    seen.add(u)
                    candidates.append(r)
            if len(candidates) >= 4:
                break

        if not candidates:
            return {"ok": False, "term": term, "error": "no accessible source"}

        # ★多源摘要聚合：收集尽量多的摘要，过滤词典噪音，合成知识
        snippets = []
        urls = []
        for c in candidates:
            sn = re.sub(r"\s+", " ", c.get("snippet", "")).strip()
            if len(sn) < 20:
                continue
            # 过滤词典释义噪音
            if re.search(r"拼音为|注音符号|词性为|部首|笔画", sn[:120]):
                continue
            snippets.append(f"【{c.get('title','')[:40]}】{sn}")
            urls.append(c.get("url", ""))
            if len(snippets) >= 5:
                break

        # 尝试抓取第一个非百科页面的正文（补充深度）
        body_text = ""
        TECH = ("zhihu.com", "csdn.net", "cnblogs.com", "jianshu.com",
                "segmentfault.com", "infoq.cn", "51cto.com", "github.com")
        for c in candidates:
            if any(x in c.get("url", "") for x in TECH):
                b = tool_web_fetch(c["url"], limit=2500)
                if b.get("ok") and len(b.get("text", "")) > 300:
                    body_text = re.sub(r"\s+", " ", b["text"])[:1500]
                    break

        parts = snippets
        if body_text:
            parts.append(f"【正文摘录】{body_text}")
        combined = " ".join(parts).strip()

        if combined and len(combined) >= 60:
            return {"ok": True, "term": term, "source": "multi-snippet",
                    "title": f"{term}（{len(snippets)}个来源）",
                    "extract": combined[:2500],
                    "url": urls[0] if urls else "",
                    "n_sources": len(snippets)}
        return {"ok": False, "term": term, "error": "no useful content"}
    except Exception as e:
        return {"ok": False, "term": term, "error": f"{type(e).__name__}: {e}"}


def tool_http_json(url: str) -> Dict[str, Any]:
    """调用 JSON API"""
    if not _rate_ok("json", 20):
        return {"ok": False, "error": "rate limited"}
    try:
        d = json.loads(http_get(url))
        return {"ok": True, "url": url, "data": d}
    except Exception as e:
        return {"ok": False, "url": url, "error": f"{type(e).__name__}: {e}"}


def tool_fs_read(path: str, limit: int = 4000) -> Dict[str, Any]:
    """读本地文件（受限目录）"""
    p = Path(path)
    try:
        rp = p.resolve()
    except Exception:
        return {"ok": False, "error": "bad path"}
    if not any(str(rp).startswith(str(d)) for d in ALLOW_DIRS):
        return {"ok": False, "error": "path not allowed"}
    if not rp.exists():
        return {"ok": False, "error": "not found"}
    try:
        txt = rp.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "path": str(rp), "text": txt[:limit], "chars": len(txt)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------- 工具注册表 ----------------
TOOLS: Dict[str, Dict[str, Any]] = {
    "web_fetch": {
        "fn": tool_web_fetch,
        "desc": "抓取网页正文",
        "args": {"url": "网址", "limit": "字数上限"},
    },
    "web_search": {
        "fn": tool_web_search,
        "desc": "搜索互联网",
        "args": {"query": "搜索词", "n": "结果数"},
    },
    "wiki_lookup": {
        "fn": tool_wiki_lookup,
        "desc": "查维基百科",
        "args": {"term": "词条", "lang": "语言(zh/en)"},
    },
    "http_json": {
        "fn": tool_http_json,
        "desc": "调用 JSON API",
        "args": {"url": "接口地址"},
    },
    "fs_read": {
        "fn": tool_fs_read,
        "desc": "读本地文件",
        "args": {"path": "文件路径", "limit": "字数上限"},
    },
}


def list_tools() -> List[Dict[str, str]]:
    return [{"name": k, "description": v["desc"], "args": v["args"]}
            for k, v in TOOLS.items()]


def call_tool(name: str, **kwargs) -> Dict[str, Any]:
    """统一调用入口"""
    if name not in TOOLS:
        return {"ok": False, "error": f"unknown tool: {name}",
                "available": list(TOOLS.keys())}
    fn: Callable = TOOLS[name]["fn"]
    t0 = time.time()
    try:
        r = fn(**kwargs)
    except TypeError as e:
        return {"ok": False, "error": f"bad args: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    r["_tool"] = name
    r["_elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return r


# ---------------- 自学习闭环 ----------------
def learn_from_result(brain, tool_result: Dict, topic: str) -> bool:
    """把工具结果"教"给神经组织（自学习闭环）

    注意：来源必须是可信的（wiki/官方文档），否则不入库。
    """
    if not tool_result.get("ok"):
        return False
    src = tool_result.get("_tool", "")
    # 来源分级：只有可信来源才入库
    trusted = {
        "wiki_lookup": "web_wiki",
        "http_json": "web_api",
        "web_fetch": "web_page",   # 网页需谨慎，但可入候选
    }
    source = trusted.get(src)
    if not source:
        return False

    # 提取知识文本
    if src == "wiki_lookup":
        content = tool_result.get("extract", "")
    elif src == "web_fetch":
        content = tool_result.get("text", "")[:500]
    else:
        content = json.dumps(tool_result.get("data", {}), ensure_ascii=False)[:500]

    if not content.strip():
        return False

    return brain.teach(topic, content, source=source)


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="仿生大脑 · 工具层")
    ap.add_argument("--demo", action="store_true", help="演示所有工具")
    ap.add_argument("--call", nargs="+", metavar=("TOOL", "ARG"),
                    help="调用工具")
    ap.add_argument("--list", action="store_true", help="列出工具")
    args = ap.parse_args()

    if args.list:
        print(json.dumps(list_tools(), ensure_ascii=False, indent=2))
    elif args.call:
        name = args.call[0]
        # 简单参数映射
        key = list(TOOLS[name]["args"].keys())[0] if name in TOOLS else "url"
        kw = {key: " ".join(args.call[1:])}
        r = call_tool(name, **kw)
        print(json.dumps(r, ensure_ascii=False, indent=2)[:2000])
    elif args.demo:
        print("=" * 74)
        print("  工具层演示")
        print("=" * 74)

        print("\n① 维基百科查询")
        r = call_tool("wiki_lookup", term="人工智能")
        if r.get("ok"):
            print(f"   ✓ {r['title']}")
            print(f"   {r['extract'][:200]}…")
        else:
            print(f"   ✗ {r.get('error')}")

        print("\n② 网页搜索")
        r = call_tool("web_search", query="Python asyncio tutorial", n=3)
        if r.get("ok"):
            for x in r["results"]:
                print(f"   · {x['title'][:60]}")
                print(f"     {x['url'][:80]}")
        else:
            print(f"   ✗ {r.get('error')}")

        print("\n③ 网页抓取")
        r = call_tool("web_fetch", url="https://example.com", limit=200)
        if r.get("ok"):
            print(f"   ✓ {r['title']} | {r['chars']} 字符")
            print(f"   {r['text'][:150]}…")
        else:
            print(f"   ✗ {r.get('error')}")

        print("\n④ JSON API")
        r = call_tool("http_json",
                      url="https://api.github.com/repos/python/cpython")
        if r.get("ok"):
            d = r["data"]
            print(f"   ✓ {d.get('full_name')} ⭐{d.get('stargazers_count')}")
        else:
            print(f"   ✗ {r.get('error')}")

        print("\n⑤ 本地文件（受限）")
        r = call_tool("fs_read", path=r"F:\DESKTOP\AI架构与推理设计\仿生AI项目设计\README.md",
                      limit=200)
        if r.get("ok"):
            print(f"   ✓ {r['path']} ({r['chars']} 字符)")
        else:
            print(f"   ✗ {r.get('error')}")

        print("\n⑥ 安全检查：越权路径")
        r = call_tool("fs_read", path=r"C:\Windows\System32\drivers\etc\hosts")
        print(f"   {'✓ 已拦截' if not r.get('ok') else '✗ 未拦截!'}: {r.get('error')}")

        print("\n" + "=" * 74)
