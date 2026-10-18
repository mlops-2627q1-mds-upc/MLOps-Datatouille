"""
Unit tests for the command-line entrypoints.
"""

import json
from unittest.mock import MagicMock, patch

from src.data.cleaning import main as cleaning_main
from src.download_dataset import main as download_main
from src.modeling.evaluation import main as eval_main
from src.modeling.selection import main as selection_main


@patch("src.data.cleaning.pd.read_csv")
@patch("src.data.cleaning.preprocess_dataframe")
@patch(
    "sys.argv",
    [
        "script",
        "--input",
        "in.csv",
        "--output",
        "out.csv",
        "--emojis-as-text",
        "true",
        "--anonymize",
        "true",
    ],
)
def test_cleaning_main(mock_preprocess, mock_read_csv):
    mock_df = MagicMock()
    mock_preprocess.return_value = mock_df
    cleaning_main()
    mock_read_csv.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_df.to_csv.assert_called_once()


@patch("src.download_dataset.download_dataset")
@patch("sys.argv", ["script", "--dataset-id", "id", "--output", "out.csv"])
def test_download_main(mock_download):
    download_main()
    mock_download.assert_called_once()


@patch("src.modeling.evaluation.mlflow")
@patch("src.modeling.evaluation.log_dataset")
@patch("src.modeling.evaluation.joblib.load")
@patch("src.modeling.evaluation.load_xy")
@patch("src.modeling.evaluation.start_model_run")
@patch("src.modeling.evaluation.save_metrics")
@patch("src.modeling.evaluation.save_top_features")
@patch("src.modeling.evaluation.compute_test_metrics")
@patch(
    "sys.argv",
    [
        "script",
        "--model",
        "m",
        "--model-path",
        "p.joblib",
        "--test",
        "t.csv",
        "--metrics-out",
        "m.json",
        "--features-out",
        "f.json",
    ],
)
def test_evaluation_main(
    mock_compute, mock_stf, mock_sm, mock_smr, mock_lxy, mock_jload, mock_ld, mock_mlflow
):
    mock_lxy.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_compute.return_value = {"test_metric": 1.0}
    eval_main()
    mock_jload.assert_called_once()
    mock_ld.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once_with({"test_metric": 1.0})
    mock_sm.assert_called_once()


@patch("src.modeling.selection.dvc.api.params_show")
@patch("src.modeling.selection.load_cv_metrics")
@patch("src.modeling.selection.select_best_model")
@patch("src.modeling.selection.build_selection")
@patch(
    "sys.argv",
    [
        "script",
        "--metric",
        "acc",
        "--metrics-dir",
        "mdir",
        "--models-dir",
        "moddir",
        "--output",
        "out.json",
    ],
)
def test_selection_main(mock_build, mock_select, mock_load, mock_dvc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_dvc.return_value = {"train": {"models": ["m1"]}}
    metrics = {"m1": {"acc": 0.9}}
    mock_load.return_value = metrics
    mock_select.return_value = ("m1", 0.9)
    mock_build.return_value = {"model": "m1"}
    selection_main()
    mock_load.assert_called_once()
    mock_select.assert_called_once_with(metrics, "acc")
    mock_build.assert_called_once()
    assert json.loads((tmp_path / "out.json").read_text()) == {"model": "m1"}
