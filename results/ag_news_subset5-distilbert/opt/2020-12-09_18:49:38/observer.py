from flask import Flask
import pandas as pd
from io import StringIO

app = Flask(__name__)

@app.route('/')
def serve_dataframe():
    df = pd.read_csv('opt_results_ag_news_subset5.csv')
    out = StringIO()
    df.to_html(out)
    return out.getvalue()

if __name__ == '__main__':
	app.run(port=6005)
