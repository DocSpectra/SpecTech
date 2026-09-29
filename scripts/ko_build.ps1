param(
    [string]$Image = "spectech-ko-specificity:36f8e835-py36-torch100-cpu"
)
$ErrorActionPreference = "Stop"
docker build --pull --tag $Image --file ko_container/Dockerfile .
docker run --rm --entrypoint python $Image -c "import torch,numpy,scipy; print(torch.__version__, numpy.__version__, scipy.__version__, torch.cuda.is_available())"
