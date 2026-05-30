from main import main


def test_main_greets(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from job-finder!" in captured.out


def test_main_returns_none():
    assert main() is None
