#以下はvscodeでやれる？？？？かも
#(notebook+sqlalchemy+flask を全て合わせてローカル可能)

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# 機械学習モデルの読込
with open('./data/KNeighbors_model.pkl', 'rb') as f:
    model = pickle.load(f)  #開いたpickleの変数格納方法
# データフレームの読込
with open('./data/df.pkl', 'rb') as f:
    df = pickle.load(f)  #開いたpickleの変数格納方法


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    # 入力値abvの取得
    param_abv = float(request.form['abv'])
    # 入力値ibuの取得
    param_ibu = float(request.form['ibu'])
    # 機械学習モデルに入力するパラメータ
    param = (param_abv, param_ibu)
    # 分類結果の取得
    pred_id = model.predict([param])[0]
    # 分類結果に該当するデータフレームの行を抽出
    pred_df = df[df["Style_id"] == pred_id].copy()
    # 入力値と分類結果の(アルコール度数, IBU)のユークリッド距離を求める
    distance = np.linalg.norm(pred_df[["abv", "ibu"]] - param, axis=1)
    # ユークリッド距離の列を追加
    pred_df['Distance'] = distance
    # データフレームを昇順でソート。上から5つの行を取得
    result_df = pred_df.sort_values(by="Distance", ascending="True")
    # 容量(ounces)違いでビールが重複することがあるので、重複を除去する
    result_df = result_df.drop_duplicates(["name"])
    # 表示するビールのスタイル名にはStyleカラムの先頭の値を代入する
    style = result_df["style"].values[0]

    return render_template('result.html',
                           param=param,
                           style=style,
                           result=result_df)
