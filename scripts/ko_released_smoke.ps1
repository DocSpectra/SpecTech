param(
    [string]$Image = "spectech-ko-specificity:36f8e835-py36-torch100-cpu",
    [string]$Volume = "spectech-ko-glove-840b-v1",
    [string]$Output = "outputs/round2/ko_reproduction/smoke"
)
$ErrorActionPreference = "Stop"
$resolved = [System.IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path "$resolved/run1", "$resolved/run2" | Out-Null
docker run --rm --volume "${Volume}:/artifacts:ro" --volume "${resolved}/run1:/output" $Image /opt/spectech/run_target.sh released-smoke
docker run --rm --volume "${Volume}:/artifacts:ro" --volume "${resolved}/run2:/output" $Image /opt/spectech/run_target.sh released-smoke
python scripts/ko_compare_predictions.py --first "$resolved/run1/predictions.txt" --second "$resolved/run2/predictions.txt" --output "$resolved/determinism.json"
