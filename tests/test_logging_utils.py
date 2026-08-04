"""Tests for logging utilities."""


class TestSetupLogging:
    """Test structured logging setup."""

    def test_setup_logging_returns_logger(self):
        import logging

        from tepid_h1.logging_utils import setup_logging

        logger = setup_logging(level="DEBUG")
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG

    def test_setup_logging_default_level(self):
        import logging

        from tepid_h1.logging_utils import setup_logging

        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_setup_logging_with_json_format(self, tmp_path):
        from tepid_h1.logging_utils import setup_logging

        log_file = tmp_path / "test.log"
        logger = setup_logging(json_format=True, log_file=log_file)
        assert logger is not None
        assert log_file.exists()

    def test_setup_logging_with_text_format(self, tmp_path):
        from tepid_h1.logging_utils import setup_logging

        log_file = tmp_path / "test.log"
        logger = setup_logging(json_format=False, log_file=log_file)
        assert logger is not None
        assert log_file.exists()


class TestLogTrainingStep:
    """Test training step logging."""

    def test_log_training_step(self, caplog):

        from tepid_h1.logging_utils import log_training_step, setup_logging

        logger = setup_logging(level="INFO")
        with caplog.at_level("INFO", logger="tepid_h1"):
            log_training_step(
                logger=logger,
                step=42,
                loss=0.5,
                gradient_norm=1.2,
                learning_rate=1e-3,
            )
        assert "Step 42" in caplog.text
        assert "loss=0.5000" in caplog.text