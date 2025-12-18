# main.py
"""
Chunked, generator-based analysis of global emissions CSV, using ONLY pandas DataFrames
for intermediate aggregation (no dict/map accumulators for analysis).

Implements three tasks:
1) Top-3 "greenest" and "dirtiest" countries by per-capita emissions over the whole period
2) Top-3 and bottom-3 countries by dispersion (variance) of yearly total emissions
3) Global totals of GDP and emissions over the observation period, plus yearly series

Each processing stage is a generator producing a stream of DataFrames:
read_csv_chunks -> select_and_clean -> task-specific reducers (via DataFrames + merges)

Charts:
- matplotlib only, one figure per chart, no seaborn, no explicit colors.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# Utilities
# ==========================
# Мы даём список «синонимов» и берём первое, что
# реально есть в файле — без чувствительности к регистру. Это упрощает «авто-детект»
def find_col(candidates: List[str], cols: List[str]) -> Optional[str]: # candidates: список возможных имён столбца 
    # cols: список реальных названий столбцов
    """Return the first matching column name (case-insensitive) from `cols`."""
    lower = {c.lower(): c for c in cols} # делаем нижний регистр для каждого имени столбца
    # cols = ["Year", "Country.Name"], то lower будет {"year": "Year" "country.name": "Country.Name"}
    for cand in candidates: # Идём по списку возможных имён, в том порядке, в котором ты их указал
        if cand.lower() in lower: # приводим к нижнему регистру и проверяем, есть ли «нормализованное» имя среди ключей словаря lower
            return lower[cand.lower()] # возвращаем оригинальное имя столбца
    return None


def merge_sum(left: pd.DataFrame,
              right: pd.DataFrame,
              on: List[str], # список имён столбцов-ключей, по которым соединяем строки
              value_cols: List[str]) -> pd.DataFrame: # какие числовые столбцы нужно просуммировать при объединении

    if left is None or len(left) == 0:
        return right.copy() # Если левый DF пустой — результатом просто будет копия правого.
    if right is None or len(right) == 0:
        return left.copy() # Если правый пустой — результатом будет копия левого.

    merged = left.merge(right, on=on, how="outer", suffixes=("_l", "_r")) # Соединяем left и right по ключам on
    for col in value_cols: # Идём по каждому числовому столбцу, который нужно суммировать
        lcol, rcol = f"{col}_l", f"{col}_r"
        if lcol not in merged: # если какая-то сторона не содержала этот столбец
            merged[lcol] = np.nan # создаём отсутствующую временную колонку
        if rcol not in merged:
            merged[rcol] = np.nan
        merged[col] = merged[lcol].fillna(0) + merged[rcol].fillna(0) # Создаём «нормальную» целевую колонку без суффиксов
        merged = merged.drop(columns=[lcol, rcol]) # итоговая сумма по этому показателю для каждойстроки-ключа.
    return merged # Возвращаем объединённый DataFrame: в нём есть все ключи из обеих таблиц и
    # просуммированные числовые столбцы из value_cols.

# ==========================
# Generators
# ==========================

def read_csv_chunks(path: str,
                    chunksize: int = 200_000,
                    **read_kwargs) -> Iterable[pd.DataFrame]: # будет принимать произвольное число пар ключ/значение
  
    for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs): # получаешь итерируемый объект (его можно перебирать в цикле), и каждая итерация даёт
        # кусок данных как DataFrame размером до chunksize строк
        yield chunk


def select_and_clean(rows: Iterable[pd.DataFrame], # входящий поток чанков
                     country_col: Optional[str], # имена нужных столбцов или None
                     year_col: Optional[str],
                     emissions_col: Optional[str],
                     pop_col: Optional[str],
                     gdp_col: Optional[str],
                     epc_col: Optional[str]) -> Iterable[pd.DataFrame]:
  
    keep = [c for c in [country_col, year_col, emissions_col, pop_col, gdp_col, epc_col] if c is not None] # Собираем список колонок, которые нужно оставить
    for df in rows: # Идём по входящим чанкам
        df = df.copy() # Делаем копию, чтобы не менять оригинал «на месте»
        if keep: # Если список нужных колонок не пуст
            have = [c for c in keep if c in df.columns] # пересечение «нужно» с тем, что реально есть в этом чанке. 
            df = df[have].copy() # Оставляем только нужные колонки
        # Приводим важные числовые колонки к числам
        for c in [emissions_col, pop_col, gdp_col, epc_col]:
            if c and c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce") # любые «плохие» значения (строки, мусор) станут NaN
        # Год приводим к числу, а потом к Int64
        if year_col and year_col in df:
            df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        yield df

#строит рейтинг стран по выбросам на душу
def compute_per_capita_rank(rows: Iterable[pd.DataFrame], # поток чанков
                            country_col: str, # имя колонки со страной
                            emissions_col: Optional[str], # имя колонки с общими выбросами
                            pop_col: Optional[str], # имя колонки с населением
                            epc_col: Optional[str]) -> pd.DataFrame: # выбросы на душу
   
    if epc_col and len(epc_col) > 0: # если колонка задана и не пустая
        acc = pd.DataFrame(columns=[country_col, "epc_sum", "epc_count"]) #создаём аккумулятор. таблицу с итогами по странам
        for df in rows: # начинаем читать поток чанков
            if epc_col not in df.columns: # защита, если в чанке нету нужной колонки
                continue
            grp = (
                df[[country_col, epc_col]]# выбирают только эти два столбца
                .dropna(subset=[epc_col]) # убирает строки, где выбросы на душу населения пустые
                .groupby(country_col, as_index=False)[epc_col] # группировка строки по стране
                .agg(["sum", "count"]) # Для каждой страны считаем сумму и количество значений на душу насел
                .reset_index() # возвращаем страну из индекса в обычную колонку
                .rename(columns={"sum": "epc_sum", "count": "epc_count"})
            )
            # вернётся новый DataFrame, где по странам просуммированы данные
            acc = merge_sum(acc, grp, on=[country_col], value_cols=["epc_sum", "epc_count"]) if not acc.empty else grp
        if acc.empty:# если все чанки без нужной колонки, то возвращается пуст табл
            return acc
        acc["per_capita"] = acc["epc_sum"] / acc["epc_count"].replace({0: np.nan}) # считаем среднее per-capita по годам для каждой страны
        result = acc[[country_col, "per_capita"]].dropna(subset=["per_capita"]) # оставляем только страну и готовый показатель
        # среднее выбросов на человека по годам
        return result.sort_values("per_capita", ascending=True) # сортируем по возрастанию


    if not emissions_col or not pop_col: # если нет имени колонки с выбросами или нет колонки с населением — посчитать нельзя
        raise ValueError("Need either emissions_per_capita column or both emissions and population columns.")
    acc = pd.DataFrame(columns=[country_col, "em_sum", "pop_sum"]) # аккумулятор с пустым дата фреймом
    for df in rows:
        need = [c for c in [country_col, emissions_col, pop_col] if c in df.columns] # проверка какие из нужных колонок есть в чанке
        if len(need) < 3: # если какой-то из колонок нет - пропуск чанка
            continue
        grp = (
            df[need] # остается только нужные три колонки
            .dropna(subset=[emissions_col, pop_col]) # убирает строки с пустыми значениями
            .groupby(country_col, as_index=False)
            .sum(numeric_only=True) # для каждой страны считаем суммы по всем числ столб
            .rename(columns={emissions_col: "em_sum", pop_col: "pop_sum"})
        )
        # объединение 
        acc = merge_sum(acc, grp, on=[country_col], value_cols=["em_sum", "pop_sum"]) if not acc.empty else grp
    if acc.empty: # # если все чанки без нужной колонки, то возвращается пуст табл
        return acc
    acc["per_capita"] = acc["em_sum"] / acc["pop_sum"].replace({0: np.nan}) # показатель на душу населения
    # на душу за весь период
    result = acc[[country_col, "per_capita"]].dropna(subset=["per_capita"])
    return result.sort_values("per_capita", ascending=True)



# считает разброс годовых выбросов по странам
def compute_emissions_dispersion(rows: Iterable[pd.DataFrame],
                                 country_col: str,
                                 year_col: str,
                                 emissions_col: str) -> pd.DataFrame:
   
    acc_cy = pd.DataFrame(columns=[country_col, year_col, "em_year_sum"])
    for df in rows:
        need = [c for c in [country_col, year_col, emissions_col] if c in df.columns]
        if len(need) < 3: # проверка есть ли в чанке нужные колонки
            continue
        grp = (
            df[need] # оставляет только нужные столбцы
            .dropna(subset=[emissions_col, year_col]) # выкидывает пустые строки
            .groupby([country_col, year_col], as_index=False)[emissions_col] #группируем по стране и году 
            .sum() # и считаем сумму выбросов в этом году для каждой страны
            .rename(columns={emissions_col: "em_year_sum"})
        )
        acc_cy = merge_sum(acc_cy, grp, on=[country_col, year_col], value_cols=["em_year_sum"]) if not acc_cy.empty else grp

    if acc_cy.empty:
        return pd.DataFrame(columns=[country_col, "n_years", "mean", "var", "std"])

    stats = ( # годовые суммы по каждой стране
        acc_cy
        .groupby(country_col, as_index=False) # Группируем по стране
        .agg(
            n_years=('em_year_sum', 'count'), # сколько лет данных есть у страны
            mean=('em_year_sum', 'mean'), # среднее годовой суммы
            var=('em_year_sum', 'var'), # дисперсия годовой суммы
            std=('em_year_sum', 'std'), # стандартное отклонение годовой суммы
        )
    )
    return stats # Возвращаем итоговую таблицу: по строке на страну, со статистиками разброса её годовых выбросов


# итоги по ВВП и выбросам (плюс годовые итоги)

def compute_totals(rows: Iterable[pd.DataFrame],
                   year_col: Optional[str],
                   emissions_col: Optional[str],
                   gdp_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    total_acc = pd.DataFrame({"dummy": [0], "em_sum": [0.0], "gdp_sum": [0.0]}) # Создаем аккумулятор для глобальных сумм 
    yearly_acc = pd.DataFrame(columns=[year_col, "emissions_total", "gdp_total"]) if year_col else None # Создаем аккумулятор для годовых сумм

    for df in rows:
        em_sum = df[emissions_col].sum(skipna=True) if emissions_col and emissions_col in df else 0.0 # Сумму выбросов
        gdp_sum = df[gdp_col].sum(skipna=True) if gdp_col and gdp_col in df else 0.0 # Сумму ВВП
        chunk_total = pd.DataFrame({"dummy": [0], "em_sum": [em_sum], "gdp_sum": [gdp_sum]})
        total_acc = merge_sum(total_acc, chunk_total, on=["dummy"], value_cols=["em_sum", "gdp_sum"]) #  Объединяем суммы из текущего чанка с общим аккумулятором 

        if year_col and year_col in df.columns: # Группируем данные по году и суммируем показатели для каждого года
            cols = [year_col] + [c for c in [emissions_col, gdp_col] if c and c in df.columns]
            if len(cols) > 1:
                grp = (
                    df[cols]
                    .groupby(year_col, as_index=False)
                    .sum(numeric_only=True)
                )
                if emissions_col and emissions_col in grp: # Переименовываем колонку выбросов или создаем нулевую, если ее нет
                    grp = grp.rename(columns={emissions_col: "emissions_total"})
                else:
                    grp["emissions_total"] = 0.0
                if gdp_col and gdp_col in grp: # Переименовываем колонку ВВП или создаем нулевую, если ее нет
                    grp = grp.rename(columns={gdp_col: "gdp_total"})
                else:
                    grp["gdp_total"] = 0.0

                keep_cols = [year_col, "emissions_total", "gdp_total"]
                grp = grp[keep_cols]
                yearly_acc = merge_sum(yearly_acc, grp, on=[year_col], value_cols=["emissions_total", "gdp_total"]) if yearly_acc is not None and not yearly_acc.empty else grp # Объединяем годовые данные с общим аккумулятором

    totals_df = pd.DataFrame({ # Создаем финальный DataFrame с общими суммами за весь период
        "total_emissions": [total_acc.at[0, "em_sum"]],
        "total_gdp": [total_acc.at[0, "gdp_sum"]],
    })
    yearly_df = yearly_acc.sort_values(year_col) if year_col and yearly_acc is not None else pd.DataFrame() # Сортируем годовые данные по возрастанию года
    return totals_df, yearly_df


# ==========================
# Построение графиков (matplotlib only)
# ==========================

def plot_per_capita(six_df: pd.DataFrame, country_col: str, out_path: str) -> None:
    fig = plt.figure()
    plt.bar(six_df[country_col], six_df["per_capita"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Emissions per capita")
    plt.title("Per-capita emissions: 3 greenest vs 3 dirtiest (all years)")
    plt.yscale('log')  
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_dispersion_with_ci(ci_df: pd.DataFrame, country_col: str, order: List[str], out_path: str) -> None:
    fig = plt.figure()
    ci_plot = ci_df.set_index(country_col).loc[order].reset_index()
    yerr = (ci_plot["mean"] - ci_plot["ci_low"], ci_plot["ci_high"] - ci_plot["mean"])
    plt.bar(ci_plot[country_col], ci_plot["mean"], yerr=yerr)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean of annual total emissions")
    plt.title("Mean annual emissions with 95% CI\n(3 smallest vs 3 largest dispersion)")
    plt.yscale('log')  
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_yearly_totals(yearly_df: pd.DataFrame, year_col: str, out_path: str) -> None:
    fig = plt.figure()
    ax1 = plt.gca()
    
    window_size = 5
    
    ax1.plot(yearly_df[year_col], yearly_df["emissions_total"], 
             label="Emissions (raw)", color='red', linewidth=1, alpha=0.3)
    
    emissions_smoothed = yearly_df["emissions_total"].rolling(window=window_size, center=True).mean()
    ax1.plot(yearly_df[year_col], emissions_smoothed, 
             label=f"Emissions ({window_size}-year MA)", color='darkred', linewidth=2)
    
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Emissions total")
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax2 = ax1.twinx()
    

    ax2.plot(yearly_df[year_col], yearly_df["gdp_total"], 
             label="GDP (raw)", color='blue', linewidth=1, alpha=0.3)
    
    # Сглаженные данные GDP
    gdp_smoothed = yearly_df["gdp_total"].rolling(window=window_size, center=True).mean()
    ax2.plot(yearly_df[year_col], gdp_smoothed, 
             label=f"GDP ({window_size}-year MA)", color='darkblue', linewidth=2)
    
    ax2.set_ylabel("GDP total")
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Легенды
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(f"Global totals over time: Emissions & GDP (with {window_size}-year moving average)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ==========================
# Обнаружение столбцов для распространенных наборов данных
# ==========================

def detect_columns(csv_path: str) -> dict:
    sample = pd.read_csv(csv_path, nrows=5)
    cols = list(sample.columns)

    country_col = find_col(["Country.Name", "country", "Country", "entity", "Country_Name"], cols)
    year_col    = find_col(["Year", "year"], cols)
    # Emissions total (CO2 total is a common proxy in this dataset)
    emissions_col = find_col([
        "Emissions.Production.CO2.Total",
        "Emissions.Total",
        "total_emissions",
        "ghg_total",
        "GHG.Total",
        "co2_total",
        "co2",
        "emissions"
    ], cols)
    pop_col = find_col(["Country.Population", "Population", "population", "pop"], cols)
    gdp_col = find_col(["Country.GDP", "GDP", "gdp", "gdp_current_usd", "gdp_usd"], cols)
    epc_col = find_col(["emissions_per_capita", "ghg_per_capita", "co2_per_capita"], cols)

    resolved = {
        "country": country_col,
        "year": year_col,
        "emissions": emissions_col,
        "population": pop_col,
        "gdp": gdp_col,
        "emissions_per_capita": epc_col
    }
    return resolved


# ==========================
# Main
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Global emissions analysis (chunked, DataFrame-only pipeline).")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--chunksize", type=int, default=250_000, help="read_csv chunksize")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cols = detect_columns(args.csv)
    country_col = cols["country"]
    year_col    = cols["year"]
    emissions_col = cols["emissions"]
    pop_col     = cols["population"]
    gdp_col     = cols["gdp"]
    epc_col     = cols["emissions_per_capita"]

    if not country_col or not year_col:
        raise RuntimeError(f"Failed to detect 'country' or 'year' columns. Detected: {cols}")
    if not emissions_col:
        raise RuntimeError(f"Failed to detect emissions column. Detected: {cols}")

    # Конвееры
    base_usecols = [c for c in [country_col, year_col, emissions_col, pop_col, gdp_col, epc_col] if c] # Создаем список колонок, которые будут использоваться в анализе
    # --- Task 1: 3 самые «зеленые» и 3 самые «грязные» страны
    stream1 = select_and_clean(
        read_csv_chunks(args.csv, chunksize=args.chunksize,
                        usecols=base_usecols,
                        dtype={country_col: "string"}),
        country_col=country_col, year_col=year_col, emissions_col=emissions_col,
        pop_col=pop_col, gdp_col=gdp_col, epc_col=epc_col
    )
    per_capita_df = compute_per_capita_rank(stream1, country_col, emissions_col, pop_col, epc_col) # Вычисляем для каждой страны средние выбросы на душу населения за весь период
    if per_capita_df.empty:
        raise RuntimeError("Per-capita result is empty. Check population/emissions columns.")

    top3_green = per_capita_df.head(3).copy()
    top3_dirty = per_capita_df.tail(3).iloc[::-1].copy()

    # --- Task 2: 3 страны с наибольшим и 3 с наименьшим разбросом суммы выбросов
    stream2 = select_and_clean(
        read_csv_chunks(args.csv, chunksize=args.chunksize,
                        usecols=[c for c in [country_col, year_col, emissions_col] if c],
                        dtype={country_col: "string"}),
        country_col=country_col, year_col=year_col, emissions_col=emissions_col,
        pop_col=None, gdp_col=None, epc_col=None
    )
    disp_df = compute_emissions_dispersion(stream2, country_col, year_col, emissions_col) # Вычисляем разброс (дисперсию) годовых выбросов для каждой страны
    disp_sorted = disp_df.sort_values("var", ascending=True)# Сортируем страны по дисперсии (var) в порядке возрастания
    disp_bottom3 = disp_sorted.head(3).copy() # Берем топ-3 стран с наименьшей дисперсией.
    disp_top3 = disp_sorted.tail(3).iloc[::-1].copy() # Берем топ-3 стран с наибольшей дисперсией и переворачиваем порядок

    # Расчет доверительного интервала для средне годовых выбросов:
    z = 1.96 # Z-значение для 95% доверительного интервала
    disp_df_ci = disp_df.copy() # Берем топ-3 стран с наибольшей дисперсией и переворачиваем порядок
    disp_df_ci["se"] = disp_df_ci["std"] / np.sqrt(disp_df_ci["n_years"]) # Вычисляем стандартную ошибку (standard error) среднего
    disp_df_ci["ci_low"] = disp_df_ci["mean"] - z * disp_df_ci["se"] # нижняя граница интервала.Рассчитываем 95% доверительный интервал для среднего значения выбросов
    disp_df_ci["ci_high"] = disp_df_ci["mean"] + z * disp_df_ci["se"] #  верхняя граница интервала

    sel_countries = pd.concat([disp_bottom3[[country_col]], disp_top3[[country_col]]], ignore_index=True) #  Создаем DataFrame только с названиями 6 выбранных стран (3 самых стабильных + 3 самых изменчивых)
    ci_six = sel_countries.merge(disp_df_ci, on=country_col, how="left") # Объединяем названия стран с их статистиками 

    # --- Task 3: Общие ВВП (GDP) и общие выбросы за период наблюдений
    # Создание пайплайна для глобальной статистики
    stream3 = select_and_clean(
        read_csv_chunks(args.csv, chunksize=args.chunksize,
                        usecols=[c for c in [year_col, emissions_col, gdp_col] if c]), # год, выбросы и ВВП
        country_col=None, year_col=year_col, emissions_col=emissions_col,
        pop_col=None, gdp_col=gdp_col, epc_col=None
    )
    totals_df, yearly_df = compute_totals(stream3, year_col=year_col, emissions_col=emissions_col, gdp_col=gdp_col)

    # --- Save tables ---
    top3_green.to_csv(os.path.join(args.out, "top3_greenest_per_capita.csv"), index=False)
    top3_dirty.to_csv(os.path.join(args.out, "top3_dirtiest_per_capita.csv"), index=False)
    disp_bottom3.to_csv(os.path.join(args.out, "bottom3_variance_emissions.csv"), index=False)
    disp_top3.to_csv(os.path.join(args.out, "top3_variance_emissions.csv"), index=False)
    ci_six.to_csv(os.path.join(args.out, "variance_ci_selected6.csv"), index=False)
    totals_df.to_csv(os.path.join(args.out, "overall_totals.csv"), index=False)
    yearly_df.to_csv(os.path.join(args.out, "yearly_totals.csv"), index=False)

    # --- Plots ---
    six_percap = pd.concat([top3_green, top3_dirty], ignore_index=True)
    plot_per_capita(six_percap, country_col, os.path.join(args.out, "plot_per_capita_top_bottom3.png"))

    order = list(disp_bottom3[country_col]) + list(disp_top3[country_col])
    plot_dispersion_with_ci(ci_six, country_col, order, os.path.join(args.out, "plot_variance_ci_top_bottom3.png"))

    if not yearly_df.empty:
        plot_yearly_totals(yearly_df, year_col, os.path.join(args.out, "plot_yearly_totals.png"))

    # --- Console summary ---
    print("=== Task 1: Per-capita (top-3 greenest) ===")
    print(top3_green.to_string(index=False))
    print("\n=== Task 1: Per-capita (top-3 dirtiest) ===")
    print(top3_dirty.to_string(index=False))

    print("\n=== Task 2: Dispersion bottom-3 (var) ===")
    print(disp_bottom3[ [country_col, "n_years", "mean", "std", "var"] ].round(3).to_string(index=False))
    print("\n=== Task 2: Dispersion top-3 (var) ===")
    print(disp_top3[ [country_col, "n_years", "mean", "std", "var"] ].round(3).to_string(index=False))

    print("\n=== Task 3: Totals ===")
    print(totals_df.to_string(index=False))
    if not yearly_df.empty:
        print(f"Years covered: {int(yearly_df[year_col].min())}–{int(yearly_df[year_col].max())}")

    print(f"\nResults saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
