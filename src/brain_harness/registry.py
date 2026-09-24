"""
HarnessPlatform · 插件注册表
==============================
把 "type 字符串" 映射到 "插件类"，让 YAML 配置能按名字组装。

用法：
    # 内置注册
    registry.register_source("file", FileSource)

    # 第三方扩展（无需改本文件）
    @registry.source("my_source")
    class MySource: ...
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Type

# 四个注册表
_SOURCES: Dict[str, type] = {}
_PROCESSORS: Dict[str, type] = {}
_POLICIES: Dict[str, type] = {}
_SINKS: Dict[str, type] = {}


def _mk_register(table: Dict[str, type], kind: str):
    def register(name: str):
        def deco(cls):
            if name in table:
                raise ValueError(f"{kind} '{name}' 已被 {table[name]} 注册")
            table[name] = cls
            return cls
        return deco
    return register


source = _mk_register(_SOURCES, "source")
processor = _mk_register(_PROCESSORS, "processor")
policy = _mk_register(_POLICIES, "policy")
sink = _mk_register(_SINKS, "sink")


def get_source(name: str) -> type:
    if name not in _SOURCES:
        raise KeyError(f"未知 source: '{name}'。已注册: {sorted(_SOURCES)}")
    return _SOURCES[name]


def get_processor(name: str) -> type:
    if name not in _PROCESSORS:
        raise KeyError(f"未知 processor: '{name}'。已注册: {sorted(_PROCESSORS)}")
    return _PROCESSORS[name]


def get_policy(name: str) -> type:
    if name not in _POLICIES:
        raise KeyError(f"未知 policy: '{name}'。已注册: {sorted(_POLICIES)}")
    return _POLICIES[name]


def get_sink(name: str) -> type:
    if name not in _SINKS:
        raise KeyError(f"未知 sink: '{name}'。已注册: {sorted(_SINKS)}")
    return _SINKS[name]


def list_all() -> Dict[str, list]:
    return {
        "sources": sorted(_SOURCES),
        "processors": sorted(_PROCESSORS),
        "policies": sorted(_POLICIES),
        "sinks": sorted(_SINKS),
    }


def build(kind: str, spec: Dict[str, Any]) -> Any:
    """按 {type: 'xxx', ...kwargs} 构造插件实例"""
    spec = dict(spec)
    t = spec.pop("type", None)
    if t is None:
        raise ValueError(f"{kind} 配置缺少 'type' 字段: {spec}")

    getters = {"source": get_source, "processor": get_processor,
               "policy": get_policy, "sink": get_sink}
    cls = getters[kind](t)
    return cls(**spec)


def load_builtins() -> None:
    """导入内置实现（触发 @register 装饰器）"""
    from . import sources as _s      # noqa: F401
    from . import processors as _p   # noqa: F401
    from . import policies as _q     # noqa: F401
    from . import sinks as _k        # noqa: F401
    from . import webui as _w        # noqa: F401  ★WebUISink
