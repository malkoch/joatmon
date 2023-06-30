from __future__ import print_function

import datetime
import json
import os
import time

import requests

from joatmon.assistant.service import BaseService
from joatmon.utility import JSONEncoder


class Service(BaseService):
    def __init__(self, api, **kwargs):
        super(Service, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
        last_time = datetime.datetime.now() - datetime.timedelta(minutes=10)
        while not self.event.is_set():
            time.sleep(0.1)

            if datetime.datetime.now() - last_time > datetime.timedelta(minutes=10):
                last_time = datetime.datetime.now()

                fields = [
                    'AO',
                    'BB.lower',
                    'BB.upper',
                    'CCI20',
                    'MACD.macd',
                    'MACD.signal',
                    'Mom',
                    'Perf.1M',
                    'Perf.3M',
                    'Perf.5Y',
                    'Perf.6M',
                    'Perf.All',
                    'Perf.W',
                    'Perf.Y',
                    'Perf.YTD',
                    'RSI',
                    'Recommend.All',
                    'Recommend.MA',
                    'Recommend.Other',
                    'SMA20',
                    'SMA200',
                    'SMA50',
                    'Stoch.D',
                    'Stoch.K',
                    'Value.Traded',
                    'Volatility.M',
                    'Volatility.W',
                    'capital_expenditures_ttm',
                    'cash_f_financing_activities_ttm',
                    'cash_f_investing_activities_ttm',
                    'cash_f_operating_activities_ttm',
                    'cash_n_short_term_invest_fq',
                    'cash_n_short_term_invest_to_total_debt_fq',
                    'change',
                    'change_abs',
                    'close',
                    'continuous_dividend_growth',
                    'continuous_dividend_payout',
                    'currency',
                    'current_ratio_fq',
                    'debt_to_equity_fq',
                    'description',
                    'dividend_payout_ratio_ttm',
                    'dividends_yield',
                    'dividends_yield_current',
                    'dps_common_stock_prim_issue_fq',
                    'dps_common_stock_prim_issue_fy',
                    'dps_common_stock_prim_issue_yoy_growth_fy',
                    'earnings_per_share_basic_ttm',
                    'earnings_per_share_diluted_ttm',
                    'earnings_per_share_diluted_yoy_growth_ttm',
                    'ebitda_ttm',
                    'enterprise_value_current',
                    'enterprise_value_ebitda_ttm',
                    'enterprise_value_to_ebit_ttm',
                    'enterprise_value_to_revenue_ttm',
                    'fractional',
                    'free_cash_flow_margin_ttm',
                    'free_cash_flow_ttm',
                    'fundamental_currency_code',
                    'gross_margin_ttm',
                    'gross_profit_ttm',
                    'logoid',
                    'market',
                    'market_cap_basic',
                    'minmov',
                    'minmove2',
                    'name',
                    'net_debt_fq',
                    'net_income_ttm',
                    'net_margin_ttm',
                    'number_of_employees_fy',
                    'oper_income_ttm',
                    'operating_margin_ttm',
                    'pre_tax_margin_ttm',
                    'price_book_fq',
                    'price_earnings_growth_ttm',
                    'price_earnings_ttm',
                    'price_free_cash_flow_ttm',
                    'price_sales_current',
                    'price_to_cash_f_operating_activities_ttm',
                    'price_to_cash_ratio',
                    'pricescale',
                    'quick_ratio_fq',
                    'research_and_dev_ratio_ttm',
                    'return_on_assets_fq',
                    'return_on_equity_fq',
                    'return_on_invested_capital_fq',
                    'sector',
                    'sell_gen_admin_exp_other_ratio_ttm',
                    'total_assets_fq',
                    'total_current_assets_fq',
                    'total_debt_fq',
                    'total_equity_fq',
                    'total_liabilities_fq',
                    'total_revenue_ttm',
                    'total_revenue_yoy_growth_ttm',
                    'type',
                    'typespecs',
                    'update_mode',
                    'volume'
                ]

                response = requests.post(
                    'https://scanner.tradingview.com/turkey/scan', data=json.dumps(
                        {
                            "columns": fields,
                            "ignore_unknown_fields": False,
                            "options": {
                                "lang": "tr"
                            },
                            "range": [
                                0, 1000
                            ],
                            "sort": {
                                "sortBy": "name",
                                "sortOrder": "asc",
                                "nullsFirst": False
                            },
                            "preset": "all_stocks"
                        }
                    )
                )
                data = json.loads(response.content.decode('utf-8'))

                stock = []

                for info in data['data']:
                    s, d = info['s'], info['d']

                    dictionary = dict(zip(fields, d))
                    dictionary['symbol'] = s
                    dictionary['date'] = last_time
                    stock.append(dictionary)

                path = os.path.join(r'X:\Cloud\OneDrive\Stock', datetime.datetime.now().strftime('%Y/%m/%d/%H-%M.json'))
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))

                open(path, 'w').write(json.dumps(stock, indent=4, cls=JSONEncoder))
