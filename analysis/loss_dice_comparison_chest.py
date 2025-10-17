from typing import Optional, List
import matplotlib.pyplot as plt

# Epoch별로 손실값과 다양한 지표(Dice, IoU 등)의 학습/검증 곡선을 시각화해주는 범용 함수입니다.

def plot_training_metrics(
    history: dict,
    tag: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    filter_mode: str = "paired",
    figsize: Optional[tuple[int, int]] = None,
    layout: str = "horizontal",
    suptitle: Optional[str] = None,
    titles: Optional[dict[str, str]] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylims: Optional[dict[str, tuple[float, float]]] = None,
    ylabel: Optional[str] = None,
    grid: bool = True,
    marker: Optional[str] = None,
    use_tight_layout: bool = True,
    tight_layout_rect: Optional[tuple[float, float, float, float]] = None
):

    """
    Epoch별 학습 및 검증지표(loss, dice, IoU 등)의 변화를 시각화하는 함수입니다.
    Segmentation, classification 등 다양한 모델의 훈련 결과를 쉽게 확인할 수 있습니다.

    ----- 매개변수 -----
    history: dict
        학습이력이 담긴 dictionary입니다. 다음과 같은 key-value 구조를 가집니다.
        예시: {'train_loss': [...], 'val_loss': [...], 'train_dice': [...], ...}

    tag: str (optional)
        그래프 제목에 덧붙일 문자열 태그입니다. (예: 모델 이름, 실험 버전 등)
        지정하지 않으면 생략됩니다.

    metrics: list of str (optional)
        시각화할 지표의 이름이 담긴 문자열 리스트입니다. 직접 지정하는 것을 권장합니다.
        문자열 순서가 시각화 그래프의 배치에 영향을 줍니다. (왼쪽에서 오른쪽, 위에서 아래 순으로 배치)
        지정하지 않으면 'train_XX', 'val_XX'와 같은 history의 key에서 추출된 지표 정보가 선택됩니다.

    filter_mode: str
        metrics가 지정되지 않은 경우, history의 key에서 지표 정보를 추출하는 방식을 결정합니다.
        지정하지 않으면 'paired'가 선택됩니다.
            - 'paired': train과 val 모두에 쌍으로 존재하는 지표만 추출
            - 'unpaired': train 또는 val 중 어느 하나에만 존재하는 지표만 추출
            - 'all': train 또는 val에 존재하는 모든 지표를 추출

    figsize: tuple of int (optional)
        전체 figsize를 결정합니다.
        지정하지 않으면 1개의 subplot을 기준으로 가로 6인치, 세로 4인치가 선택되며,
        metrics 및 layout에 따라 가로 또는 세로가 배수로 조정됩니다.

    layout: str
        시각화 그래프를 배치하는 방식을 결정합니다.
        지정하지 않으면 'horizontal'이 선택됩니다.
            - 'horizontal': 그래프가 가로 방향으로 한 줄에 배치됩니다.
            - 'vertical': 그래프가 세로 방향으로 한 줄에 배치됩니다.

    suptitle: str (optional)
        전체 그래프(Figure)의 제목으로 표시할 문자열입니다.
        지정하지 않으면 기본 제목("Training Metrics History")이 사용되며,
        tag가 주어질 경우 뒤에 덧붙여집니다.

    titles: dict of (str: str) (optional)
        각 metric별 subplot에 표시할 제목을 담은 dictionary입니다.
        key는 지표 이름(metric)이며, value는 subplot의 제목 문자열입니다.

        - titles가 주어지고, 해당 metric이 key로 존재하면: titles[metric] 값을 subplot 제목으로 사용합니다.
        - titles가 주어졌지만 해당 metric이 key로 존재하지 않으면: metric 문자열 자체를 제목으로 사용합니다.
        - titles가 주어지지 않고 metrics가 명시된 경우: metric 문자열을 그대로 subplot 제목으로 사용합니다.
        - titles가 주어지지 않고 metrics가 None인 경우(자동 추출된 경우): metric.capitalize()를 제목으로 사용합니다.

    xlim: dict of (str: tuple of float) (optional)
        모든 subplot에 동일하게 적용할 x축의 범위입니다.
        예: (0, 50). 지정하지 않으면 matplotlib가 자동으로 범위를 설정합니다.

    ylims: dict of (str: tuple of float) (optional)
        Key는 각 지표의 이름이며, 각 지표에 적용할 y축의 범위(y_min, y_max)를 value로 가지는 딕셔너리입니다.
        subplot마다 다른 y축 범위를 설정하고자 할 때 사용합니다.
        예: {'loss': (0.0, 2.0), 'dice': (0.6, 1.0)}
        지정하지 않으면 matplotlib가 자동으로 축 범위를 결정합니다.

        주의: metrics를 지정하지 않고 자동으로 추출하는 경우(metrics=None),
        사용자가 지정한 ylims은 무시됩니다.
        축 범위를 설정하려면 metrics를 명시적으로 지정해 주세요.

    ylabel: str (optional)
        모든 subplot의 y축 라벨로 사용할 공통 문자열입니다.
        지정하지 않으면 각 subplot별로 해당 지표 이름(metric)을 기반으로 자동 설정됩니다.

    grid: bool
        각 subplot에 격자(grid)를 표시할지 여부를 결정합니다.
        지정하지 않으면 True가 선택되어 격자가 표시됩니다.

    marker: str (optional)
        학습 및 검증 곡선에 점 형태의 marker를 표시합니다.
        지정하지 않으면 matplotlib의 기본 스타일로 선만 그려집니다.
        예: 'o' (원), 's' (사각), '^' (삼각), '.' (점), None (선만 표시)

    use_tight_layout: bool
        subplot 간 간격을 자동으로 정리하는 matplotlib의 tight_layout 기능 사용 여부입니다.
        False로 설정하면 수동으로 레이아웃을 조정해야 합니다.

    tight_layout_rect: tuple of float (optional)
        tight_layout 적용 시 사용할 영역 범위를 지정하는 rect 값입니다.
        (left, bottom, right, top) 형식의 0~1 비율 tuple이며,
        지정하지 않으면 matplotlib의 기본 범위인 (0, 0, 1, 1)를 사용합니다.

    ----- 경고메세지 -----
    - history의 각 key에 저장된 value의 개수가 서로 다르면 이를 알리는 경고메세지 및
      각 key에 저장된 value의 개수가 출력됩니다.
    - metrics에 지정되었으나, history의 key에는 존재하지 않는 지표가 확인된 경우 경고메세지가 출력됩니다.
    - metrics에 어떠한 지표도 선택되지 않은 경우, 경고메세지를 출력하고 실행 중인 함수를 즉시 종료합니다.

    ----- 반환값 -----
    이 함수는 matplotlib 그래프를 출력하며, 별도로 값을 반환하지 않습니다.
    """

    # history의 각 key에 저장된 value의 개수를 확인합니다.
    lengths = {k: len(v) for k, v in history.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        print("⚠️ Warning: Metrics in history have different lengths!")
        for k, l in lengths.items():
            print(f"  - {k}: {l}")
        print()
    
    # metrics가 지정된 경우, 모든 metrics가 history의 key에 존재하는지 확인합니다.
    if metrics is not None:
        valid_metrics = [m for m in metrics if f"train_{m}" in history or f"val_{m}" in history]
        if not valid_metrics:
            print("⚠️ Warning: None of the specified metrics are present in history.")
            return
        metrics = valid_metrics

    # metrics가 지정되지 않은 경우, filter_mode에서 지정한 방식에 따라 history의 key에서 지표를 추출합니다.
    # metrics가 지정되지 않은 경우, ylim에 지정된 지표별 최소 및 최대값은 무시됩니다.
    if metrics is None:
        all_keys = set(history.keys())
        train_keys = [k for k in all_keys if k.startswith("train_")]
        val_keys   = [k for k in all_keys if k.startswith("val_")]

        train_metrics = set(k.split("_", 1)[1] for k in train_keys)
        val_metrics   = set(k.split("_", 1)[1] for k in val_keys)

        if filter_mode == "paired":
            metrics = list(train_metrics & val_metrics)
        elif filter_mode == "unpaired":
            metrics = list((train_metrics | val_metrics) - (train_metrics & val_metrics))
        elif filter_mode == "all":
            metrics = list(train_metrics | val_metrics)
        else:
            raise ValueError("Filter_mode must be 'paired', 'unpaired', or 'all'")

        ylims = None

    # history의 key에서 지표를 추출할 수 없는 경우, 함수가 종료됩니다.
    if not metrics:
        print("⚠️ No metrics found to plot. The 'plot_training_metrics' function has been terminated.")
        return

    # 1부터 'history의 key에 저장된 value의 개수'까지의 자연수를 Epoch 번호의 리스트로 저장합니다.
    train_keys = [k for k in history if k.startswith("train_")]
    epochs = range(1, len(history[train_keys[0]]) + 1)

    # Layout에 따라 subplot의 배열을 구성하고 전체 figsize를 결정합니다.
    n = len(metrics)
    if layout == "horizontal":
        nrows, ncols = 1, n
        default_figsize = (6 * n, 4)
    elif layout == "vertical":
        nrows, ncols = n, 1
        default_figsize = (6, 4 * n)
    else:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    # 사용자가 지정한 figsize가 있으면, 그것을 우선 사용합니다.
    plt.figure(figsize=figsize or default_figsize)

    # 각 지표별 subplot을 생성합니다.
    for i, metric in enumerate(metrics, 1):
        plt.subplot(nrows, ncols, i)
        t_key, v_key = f"train_{metric}", f"val_{metric}"

        if t_key in history:
            if marker is not None:
                plt.plot(epochs, history[t_key], label=f"Train {metric}", marker=marker)
            else:
                plt.plot(epochs, history[t_key], label=f"Train {metric}")

        if v_key in history:
            if marker is not None:
                plt.plot(epochs, history[v_key], label=f"Val {metric}", marker=marker)
            else:
                plt.plot(epochs, history[v_key], label=f"Val {metric}")

        if titles:
            subplot_title = titles.get(metric, metric)
        elif metrics is not None:
            subplot_title = metric
        else:
            subplot_title = metric.capitalize()
        plt.title(subplot_title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel if ylabel else metric.capitalize())
        if xlim:
            plt.xlim(xlim)
        if ylims and metric in ylims:
            plt.ylim(ylims[metric])     
        plt.legend()
        plt.grid(grid)

    base_title = suptitle or "Training Metrics History"
    final_title = f"{base_title} — {tag}" if tag else base_title
    plt.suptitle(final_title)
    if use_tight_layout:
        if tight_layout_rect:
            plt.tight_layout(rect=tight_layout_rect)
        else:
            plt.tight_layout()
    plt.show()
