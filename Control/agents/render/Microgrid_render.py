import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd


index = pd.date_range('2010-01-01', periods=365, freq='H')


def date2num(date):
  converter = mdates.datestr2num(date)
  return converter(date)

class Microgrid_render:
    def __init__(self,title=None):
        fig=plt.figure()
        fig.suptitle(title)

        self.net_demand_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2,colspan=1)

        self.battery_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1,  sharex=self.net_demand_ax)

        self.hydrogen_ax = self.battery_ax.twinx()

        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top = 0.90, wspace = 0.2, hspace = 0)
        plt.show(block=False)

    def render(self, current_step, net_demand, trades, window_size=24):
        self.net_demands[current_step] = net_demand
        window_start = max(current_step - window_size, 0)
        step_range = range(window_start,
                           current_step + 1)  # Format dates as timestamps, necessary for candlestick graph
        index = pd.date_range('2010-01-01', periods=365, freq='H')
        dates = np.array(mdates.datestr2num(str(index)))

        self._render_net_demand(current_step, net_demand, window_size,
                               dates)
        self._render_price(current_step, net_demand, dates, step_range)
        self._render_volume(current_step, net_demand, dates, step_range)
        self._render_trades(current_step, trades, step_range)  # Format the date ticks to be more easily read
        self.price_ax.set_xticklabels(index.values[step_range],
                                      rotation=45, horizontalalignment='right')  # Hide duplicate net worth date labels
        plt.setp(self.net_demand_ax.get_xticklabels(),
                 visible=False)  # Necessary to view frames before they are unrendered
        plt.pause(0.001)


    def _render_net_demand(self, current_step, net_demand, step_range,
                          dates):
        # Clear the frame rendered last step
        self.net_demand_ax.clear()  # Plot net worths
        self.net_demand_ax.plot_date(dates, self.net_demands[step_range], '-', label='NetWorth')  # Show legend, which uses the label we defined for the plot above
        self.net_demand_ax.legend()
        legend = self.net_demand_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)
        last_date = date2num(self.df['Date'].values[current_step])
        last_net_demand = self.net_demands[current_step]  # Annotate the current net worth on the net worth graph
        self.net_demand_ax.annotate('{0:.2f}'.format(net_demand),
                                   (last_date, last_net_demand),
                                   xytext=(last_date, last_net_demand),
                                   bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")  # Add space above and below min/max net worth
        self.net_demand_ax.set_ylim(
            min(self.net_demands[np.nonzero(self.net_demands)]) / 1.25,
            max(self.net_demands) * 1.25)