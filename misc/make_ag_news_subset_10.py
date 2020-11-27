from src.utils.datasets import make_ag_news_subset

if __name__ == '__main__':
    make_ag_news_subset('../datasets/ag_news_subset10.csv', subset_size=0.1)