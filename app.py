from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import jpholiday
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


APP_TITLE = "処方箋枚数予測アプリ"
MODEL_PATH = Path("prescription_model.joblib")
FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "day",
    "is_holiday",
    "is_month_start",
    "is_month_end",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
]


def read_uploaded_csv(uploaded_file: BytesIO) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    required_columns = {"date", "prescription_count"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSVに必要な列がありません: {missing_text}")

    df = df[["date", "prescription_count"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["prescription_count"] = pd.to_numeric(df["prescription_count"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        raise ValueError("有効な日付データがありません。")

    df = (
        df.groupby("date", as_index=False)["prescription_count"]
        .sum(min_count=1)
        .sort_values("date")
    )
    return df


def clean_daily_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    messages: list[str] = []
    original_rows = len(df)
    start_date = df["date"].min()
    end_date = df["date"].max()

    daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
    daily_df = df.set_index("date").reindex(daily_index)
    daily_df.index.name = "date"

    missing_count = daily_df["prescription_count"].isna().sum()
    if missing_count:
        daily_df["prescription_count"] = (
            daily_df["prescription_count"].interpolate(method="linear").ffill().bfill()
        )
        messages.append(f"欠損または未入力の日付 {missing_count} 件を補完しました。")

    daily_df["prescription_count"] = daily_df["prescription_count"].round().astype(int)
    daily_df = daily_df.reset_index()

    if len(daily_df) != original_rows:
        messages.append("日付の重複集計または日次データへの整形を行いました。")

    return daily_df, messages


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    dates = featured["date"]
    featured["day_of_week"] = dates.dt.dayofweek
    featured["month"] = dates.dt.month
    featured["day"] = dates.dt.day
    featured["is_holiday"] = dates.dt.date.map(lambda value: int(jpholiday.is_holiday(value)))
    featured["is_month_start"] = dates.dt.is_month_start.astype(int)
    featured["is_month_end"] = dates.dt.is_month_end.astype(int)
    return featured


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_calendar_features(df)
    count = featured["prescription_count"]
    featured["lag_7"] = count.shift(7)
    featured["lag_14"] = count.shift(14)
    featured["lag_28"] = count.shift(28)
    featured["rolling_mean_7"] = count.shift(1).rolling(window=7).mean()
    featured["rolling_mean_14"] = count.shift(1).rolling(window=14).mean()
    return featured


def build_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    featured = add_time_series_features(df)
    model_df = featured.dropna(subset=FEATURE_COLUMNS + ["prescription_count"]).copy()
    x = model_df[FEATURE_COLUMNS]
    y = model_df["prescription_count"]
    return x, y


def train_model(x: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def make_future_features(date: pd.Timestamp, history: pd.DataFrame) -> pd.DataFrame:
    history_by_date = history.set_index("date")["prescription_count"]

    def lag(days: int) -> float:
        target_date = date - pd.Timedelta(days=days)
        return float(history_by_date.loc[target_date])

    past_14 = history[history["date"] < date].tail(14)["prescription_count"]
    row = {
        "day_of_week": date.dayofweek,
        "month": date.month,
        "day": date.day,
        "is_holiday": int(jpholiday.is_holiday(date.date())),
        "is_month_start": int(date.is_month_start),
        "is_month_end": int(date.is_month_end),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "lag_28": lag(28),
        "rolling_mean_7": float(past_14.tail(7).mean()),
        "rolling_mean_14": float(past_14.mean()),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def forecast(model: RandomForestRegressor, history: pd.DataFrame, days: int) -> pd.DataFrame:
    working_history = history[["date", "prescription_count"]].copy()
    predictions = []
    next_date = working_history["date"].max() + pd.Timedelta(days=1)

    for offset in range(days):
        forecast_date = next_date + pd.Timedelta(days=offset)
        features = make_future_features(forecast_date, working_history)
        predicted_count = max(0, round(float(model.predict(features)[0])))
        predictions.append(
            {
                "date": forecast_date,
                "predicted_prescription_count": predicted_count,
            }
        )
        working_history = pd.concat(
            [
                working_history,
                pd.DataFrame(
                    [
                        {
                            "date": forecast_date,
                            "prescription_count": predicted_count,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


def plot_forecast(history: pd.DataFrame, prediction_30d: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        history["date"],
        history["prescription_count"],
        label="過去データ",
        color="#2563eb",
        linewidth=2,
    )
    ax.plot(
        prediction_30d["date"],
        prediction_30d["predicted_prescription_count"],
        label="予測データ",
        color="#dc2626",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("日付")
    ax.set_ylabel("処方箋枚数")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    export_df = df.copy()
    export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")
    return export_df.to_csv(index=False).encode("utf-8-sig")


def model_to_bytes(model: RandomForestRegressor) -> bytes:
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def render_prediction_results(history: pd.DataFrame, model: RandomForestRegressor) -> None:
    prediction_7d = forecast(model, history, 7)
    prediction_30d = forecast(model, history, 30)

    st.subheader("予測グラフ")
    st.pyplot(plot_forecast(history, prediction_30d))

    st.subheader("7日予測表")
    st.dataframe(prediction_7d, use_container_width=True)

    st.subheader("30日予測表")
    st.dataframe(prediction_30d, use_container_width=True)

    st.download_button(
        label="30日予測CSVをダウンロード",
        data=dataframe_to_csv_bytes(prediction_30d),
        file_name="prescription_count_forecast_30d.csv",
        mime="text/csv",
    )

    st.download_button(
        label="学習済みモデルをダウンロード",
        data=model_to_bytes(model),
        file_name="prescription_model.joblib",
        mime="application/octet-stream",
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    if uploaded_file is None:
        st.info("date,prescription_count 形式のCSVをアップロードしてください。")
        return

    try:
        raw_df = read_uploaded_csv(uploaded_file)
        history_df, clean_messages = clean_daily_data(raw_df)
    except ValueError as exc:
        st.error(str(exc))
        return

    for message in clean_messages:
        st.warning(message)

    if len(history_df) < 60:
        st.warning("データが少ないため、予測精度が不安定になる可能性があります。60日以上のデータを推奨します。")

    st.subheader("過去データのプレビュー")
    st.dataframe(history_df.head(30), use_container_width=True)

    if st.button("学習実行", type="primary"):
        x, y = build_training_data(history_df)
        if len(x) < 10:
            st.error("学習に使えるデータが不足しています。少なくとも40日以上の日次データを用意してください。")
            return

        with st.spinner("モデルを学習し、予測を作成しています。"):
            model = train_model(x, y)
            joblib.dump(model, MODEL_PATH)
            st.success(f"学習が完了しました。学習データ件数: {len(x)} 件")
            render_prediction_results(history_df, model)


if __name__ == "__main__":
    main()
