#прогнозирование минимальной температуры (вариант 2)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

#заголовок приложения
st.set_page_config(page_title="Прогноз диабета", layout="wide")
st.title("Прогнозирование прогрессии диабета")
st.markdown("Лабораторная работа №8. Демонстрация моделей машинного обучения.")
st.markdown("Выполнила: Лютикова А. А.")
st.markdown("Группа: ИБМ3-64Б")
st.markdown("---")

#загрузка данных
@st.cache_data
def load_data():
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    X_df = data['data']
    y_series = data['target']
    return X_df, y_series

data_load_state = st.text('Загрузка данных...')
X_df, y_series = load_data()
data_load_state.text('Данные загружены!')

#подготовка признаков и разделение
X = X_df.values
y = y_series.values
feature_names = list(X_df.columns)

#разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#информация о данных
st.header("Информация о данных")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего записей", f"{len(X)}")
col2.metric("Обучающая выборка", f"{len(X_train)}")
col3.metric("Тестовая выборка", f"{len(X_test)}")
col4.metric("Признаков", f"{len(feature_names)}")

col_left, col_right = st.columns(2)
with col_left:
    if st.checkbox('Показать первые 10 строк'):
        st.dataframe(X_df.head(10))
with col_right:
    if st.checkbox('Показать статистические показатели'):
        st.dataframe(X_df.describe())

#корреляция
if st.checkbox('Показать корреляционную матрицу'):
    st.subheader("Корреляционная матрица")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    df_with_target = X_df.copy()
    df_with_target['target'] = y
    sns.heatmap(df_with_target.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax_corr)
    st.pyplot(fig_corr)

st.markdown("---")

#боковая панель выбор моделей и гиперпараметров
st.sidebar.header('Настройка моделей')

models_list = ['KNN', 'Ridge', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVR']
models_select = st.sidebar.multiselect(
    'Выберите модели для обучения',
    models_list,
    default=['Random Forest']
)

#гиперпараметры
params = {}
st.sidebar.markdown("---")
st.sidebar.subheader("Гиперпараметры")

if 'KNN' in models_select:
    st.sidebar.markdown("**KNN**")
    params['n_neighbors'] = st.sidebar.slider('Количество соседей', 1, 30, 5, 1, key='knn')

if 'Ridge' in models_select:
    st.sidebar.markdown("**Ridge**")
    params['alpha'] = st.sidebar.slider('Alpha (регуляризация)', 0.01, 10.0, 1.0, 0.01, key='ridge')

if 'Decision Tree' in models_select:
    st.sidebar.markdown("**Decision Tree**")
    params['max_depth'] = st.sidebar.slider('Максимальная глубина', 1, 30, 5, 1, key='dt')

if 'Random Forest' in models_select:
    st.sidebar.markdown("**Random Forest**")
    params['rf_n_estimators'] = st.sidebar.slider('Число деревьев', 10, 200, 100, 10, key='rf_n')
    params['rf_max_depth'] = st.sidebar.slider('Макс. глубина', 1, 30, 10, 1, key='rf_d')

if 'Gradient Boosting' in models_select:
    st.sidebar.markdown("**Gradient Boosting**")
    params['gb_n_estimators'] = st.sidebar.slider('Число деревьев', 10, 200, 100, 10, key='gb_n')
    params['gb_learning_rate'] = st.sidebar.slider('Темп обучения', 0.01, 1.0, 0.1, 0.01, key='gb_lr')

if 'SVR' in models_select:
    st.sidebar.markdown("**SVR**")
    params['svr_C'] = st.sidebar.slider('C (регуляризация)', 0.1, 10.0, 1.0, 0.1, key='svr_c')
    params['svr_epsilon'] = st.sidebar.slider('Epsilon', 0.01, 1.0, 0.1, 0.01, key='svr_e')

st.sidebar.markdown("---")

#кнопка запуска обучения
if st.sidebar.button('Обучить модели', use_container_width=True):

    st.header("Результаты обучения")

    with st.spinner('Идёт обучение моделей...'):
        results = {}

        for model_name in models_select:
            if model_name == 'KNN':
                model = KNeighborsRegressor(n_neighbors=params['n_neighbors'])
            elif model_name == 'Ridge':
                model = Ridge(alpha=params['alpha'])
            elif model_name == 'Decision Tree':
                model = DecisionTreeRegressor(max_depth=params['max_depth'], random_state=0)
            elif model_name == 'Random Forest':
                model = RandomForestRegressor(
                    n_estimators=params['rf_n_estimators'],
                    max_depth=params['rf_max_depth'],
                    random_state=0
                )
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingRegressor(
                    n_estimators=params['gb_n_estimators'],
                    learning_rate=params['gb_learning_rate'],
                    random_state=0
                )
            elif model_name == 'SVR':
                model = SVR(C=params['svr_C'], epsilon=params['svr_epsilon'])

#обучение
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

#метрики
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results[model_name] = {
                'model': model,
                'y_pred': y_pred,
                'rmse': rmse,
                'mae': mae
            }


#метрики в карточках
    st.subheader("Метрики качества")

    cols = st.columns(len(results))
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.metric(
                label=f"**{name}**",
                value=f"RMSE: {res['rmse']:.2f}",
                delta=f"MAE: {res['mae']:.1f}"
            )

#таблица результатов
    st.markdown("---")
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            'Модель': name,
            'RMSE': f"{res['rmse']:.4f}",
            'MAE': f"{res['mae']:.4f}"
        })


#график: фактические и прогнозные значения
    st.markdown("---")
    st.subheader("Сравнение прогнозов (первые 100 записей теста)")

    fig, ax = plt.subplots(figsize=(14, 6))

#реальные значения
    ax.plot(y_test[:100], label='Фактические значения', color='black', linewidth=2, zorder=5)

#прогнозы моделей
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5', '#A5A5A5']
    for i, (name, res) in enumerate(results.items()):
        ax.plot(res['y_pred'][:100], label=f"{name} (RMSE={res['rmse']:.1f})",
                color=colors[i % len(colors)], alpha=0.85, linewidth=1.2)

    ax.set_xlabel('Номер наблюдения в тестовой выборке')
    ax.set_ylabel('Прогрессия диабета')
    ax.set_title('Сравнение прогнозов моделей машинного обучения')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


#столбчатая диаграмма сравнения rmse
    st.markdown("---")
    st.subheader("Сравнение моделей по RMSE")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    rmse_values = [results[n]['rmse'] for n in names]
    colors_sorted = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5', '#A5A5A5']

#cортировка от лучшего к худшему
    sorted_idx = np.argsort(rmse_values)
    names_sorted = [names[i] for i in sorted_idx]
    rmse_sorted = [rmse_values[i] for i in sorted_idx]

    bars = ax2.bar(names_sorted, rmse_sorted,
                   color=[colors_sorted[i] for i in sorted_idx], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('RMSE')
    ax2.set_title('Сравнение качества моделей (от лучшей к худшей)')

    for bar, val in zip(bars, rmse_sorted):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontweight='bold', fontsize=11)

    st.pyplot(fig2)


#важность
    tree_models = {n: r for n, r in results.items()
                   if hasattr(r['model'], 'feature_importances_')}

    if tree_models:
        st.markdown("---")
        st.subheader("Важность признаков (для древовидных моделей)")

        n_tree_models = len(tree_models)
        fig3, axes3 = plt.subplots(1, n_tree_models, figsize=(6 * n_tree_models, 5))
        if n_tree_models == 1:
            axes3 = [axes3]

        for ax, (name, res) in zip(axes3, tree_models.items()):
            importances = res['model'].feature_importances_
            sorted_idx_imp = np.argsort(importances)
            ax.barh([feature_names[i] for i in sorted_idx_imp],
                    importances[sorted_idx_imp],
                    color='#4472C4', edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Важность')
            ax.set_title(name)

        st.pyplot(fig3)

st.sidebar.markdown('---')
st.sidebar.info('Выберите модели, настройте гиперпараметры и нажмите **"Обучить модели"**')