import pytest


def test_environment():
    """Basic test to verify the testing environment works."""
    assert True


def test_imports():
    """Test that our main packages can be imported."""
    try:
        import src.data_ingestion
        import src.data_processing
        import src.analytics
        import src.strategies
        import src.backtesting
        import src.live_trading
        import src.utils
        assert True
    except ImportError as e:
        pytest.fail(f"Import error: {e}")


def test_config_loading():
    """Test loading configuration file."""
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert 'environment' in config
        assert 'data_sources' in config
    except Exception as e:
        pytest.fail(f"Config loading error: {e}") 