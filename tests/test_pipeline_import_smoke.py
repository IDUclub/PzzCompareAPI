import importlib


def test_pipeline_module_import_succeeds() -> None:
    module = importlib.import_module("pipeline_modules.pipeline_v25")
    assert module is not None


def test_business_pipeline_impl_import_succeeds() -> None:
    module = importlib.import_module("pipeline_modules.business.pipeline_impl")
    assert module is not None
