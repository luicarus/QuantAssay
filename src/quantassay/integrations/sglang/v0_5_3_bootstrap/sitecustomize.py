import os

backend = os.environ.get("KERNSCOPE_SGLANG_BACKEND")
if backend:
    from quantassay.integrations.sglang.v0_5_3 import install

    install(backend)
