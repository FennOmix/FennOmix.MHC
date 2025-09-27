from fennomix_mhc import __github__, __license__, __version__


def test_package_metadata():
    assert isinstance(__version__, str) and len(__version__) > 0
    assert __github__.startswith("https://")
    assert "Apache" in __license__
