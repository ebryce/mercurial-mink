import numpy as np
import pandas as pd
import math
import logging

logging.basicConfig(level=logging.error)

class Market:
    def __init__(self, randomize_environment=False):

        self.lob = LOB(asset=Asset())
        self.agents = [MarketMaker()]

        if randomize_environment:
            self._randomize_book(n_orders=15)

    def place_an_order(self, order, lob, agent):
        trades = lob.send(order)
        agent.book_an_order(order, trades)

    def evolve(self):
        '''Evolves the market by one time step by having each agent submit an order and then evolving the order book'''
        for agent in self.agents:
            order = agent.evolve(self)
            if order is not None:
                trades = self.lob.send(order)
            agent._purge()
        self.fv_process.evolve()
        return trades

    def _randomize_book(self, bid_skew=0.5, n_orders=10, max_depth=5, min_spread=0.05):
        
        fair_value = self.fv_process.fv
        logging.info('Randomizing the order book with {} orders around {:,.2f}; spread {:.2f}-{:.2f}'.format(
            n_orders, fair_value, min_spread, min_spread+2*max_depth*self.lob.tick_sz)
        )
        
        trades = []
        for _ in range(n_orders):
            order = self._random_order(
                fair_value=fair_value, bid_skew=bid_skew,min_spread=min_spread,
                max_depth=max_depth, marketable=False)
            trade = self.lob.send(order)
            trades.extend(trade)

        return trades

    def _random_order(self, fair_value=None, side=None, price=None, size=None, min_spread=0.05,
                      bid_skew=0.5, marketable_skew=0.5, max_depth=5, marketable=None):
        
        fair_value = self.fv_process.fv if fair_value is None else fair_value

        if side is None:
            side = 'B' if np.random.rand() < bid_skew else 'S'

        if size is None:
            size = np.random.randint(1, max_depth) * self.lob.lot_sz

        if price is None:
            if marketable is None:
                marketable = np.random.rand() < marketable_skew
                marketable = False if (side=='B' and np.isnan(self.lob.get_bid())) or (side=='S' and np.isnan(self.lob.get_offer())) else marketable

            if marketable:
                # If the order is marketable, simply cross the spread
                price = self.lob.get_offer() if side=='B' else self.lob.get_bid()
                logging.debug('Random order to {} {} is marketable; will attempt to trade at ${:,.2f} (mkt={:,.2f}/{:,.2f})'.format(side, size, price, self.lob.get_bid(), self.lob.get_offer(),))
            else:
                # If the order is non-marketabe, price it randomly near the quote
                if side=='B':
                    if np.isnan(self.lob.get_bid()):
                        logging.debug('No bid available, pricing at fair value {:,.4f} minus half the minimum spread {:,.2f}'.format(fair_value, min_spread))
                        price = self.lob.round_to_tick(fair_value - min_spread*0.5)
                    else:
                        price = self.lob.round_to_tick(self.lob.get_bid() - self.lob.tick_sz*2*max_depth*np.random.rand())
                else:
                    if np.isnan(self.lob.get_offer()):
                        logging.debug('No offer available, pricing at fair value {:,.4f} plus half the minimum spread {:,.2f}'.format(fair_value, min_spread))
                        price = self.lob.round_to_tick(fair_value + min_spread*0.5)
                    else:
                        price = self.lob.round_to_tick(self.lob.get_offer() + self.lob.tick_sz*2*max_depth*np.random.rand())
                logging.info('Random passive order to {} {} at ${:,.2f} joining {} (mkt={:,.2f}/{:,.2f})'.format(side, size, price, self.lob.get_depth(price), self.lob.get_bid(), self.lob.get_offer()))

        order = Order(side=side, size=size, price=price, marketable=marketable)
        return order

class Asset:
    '''Describes the movement of the fair value of the risky asset; looks a bit like Kyle (1985)'''

    def __init__(self, drift=0, variance=1, initial_state=10):
        self.μ, self.σ = drift, variance
        self.F = initial_state # The final payoff of the risky asset
        self.midpoint = initial_state
        return None
    
    def evolve(self):
        ''' The fair value of the risky asset follows a martingale process '''
        self.F = self.F + self.σ*np.random.randn()
        return self.F
    
    def _get_prediction(self, σ_η):
        ''' Return the prediction of the fair value of the risky asset '''
        η = np.random.normal(0, σ_η) # The noise in the prediction
        s = self.F + η               # A signal based on the noisy view of FV
        return s
    
    def get_forecast(self, σ_η):
        ''' Return the forecast of the fair value of the risky asset 
        E[F|s] = m + β (s - m)'''
        β = self.σ**2 / (self.σ**2 - σ_η**2)
        EF = self.midpoint + β*(self._get_prediction(σ_η) - self.midpoint)
        return EF

class Agent:
    def __init__(self, noise=None):
        ''' An agent that can interact with the market '''
        self.orders = []
        self.σ_η = noise
        return None

    def _purge(self, aggressive=False):
        for order in self.orders:
            if (order.size_remaining == 0) or aggressive:
                self.orders.remove(order)

    def book_an_order(self, order, trades):
        self.orders.append(order)
        return order
    
    def determine_fv(self, asset):
        ''' Determine the agent's view of the fair value of the risky asset '''
        return asset.get_forecast(self.σ_η)

    def evolve(self, market):
        if self.balance < self.risk_threshold:
            order = self.generate_an_order(market=market)
            self.book_an_order(order=order)
        else:
            self._purge(aggressive=True)
        return order

class LiquidityTrader(Agent):
    '''A trader who generally takes liquidity and is willing to pay the spread; they may or may not be informed'''

    def __init__(self, information=0.5):
        '''Information- the probability that the agent is informed'''
        Agent.__init__(self)
        return None

class InformedTrader(Agent):
    '''A trader who possesses some information about the fair value of the asset'''
    def __init__(self):
        return None

class MarketMaker(Agent):
    '''A market maker who posts bids and offers with the intent of earning a spread
    Will later start to look a bit like Stoll (1978) and Grossman & Miller (1988)'''
    def __init__(self):
        return None

class Order:
    def __init__(self, side, size, price=None, marketable=False):
        '''Create a new order'''
        self.side = side
        self.size = size
        self.limit = price
        self.marketable = marketable
        self.size_filled = 0
        self.size_remaining = self.size - self.size_filled
        self.notional_traded = 0
        self.prints = []
        return None

    def give_fill(self, size, price):
        '''Gives a fill to the order, reducing its size remaining and increasing its notional traded'''
        if size>self.size_remaining:
            logging.warning('Fill {} would exceed remaining order size {}'.format(size, self.size_remaining))
            size = min(size, self.size_remaining)
        
        self.notional_traded = self.notional_traded + size * price
        self.size_filled = self.size_filled + size
        self.size_remaining = max(0,self.size - self.size_filled)


class LOB:
    def __init__(self, asset):
        '''Initiaize an empty limit order book'''
        self.offers = {}
        self.bids = {}
        self.prints = []
        self.asset = asset

        self.tick_sz = 0.01
        self.lot_sz = 100

        return None

    def _purge(self):
        '''Remove all filled orders from the book to avoid memory accumulating'''
        orders_purged, orders_remaining = 0, 0

        for level in self.offers:
            for order in self.offers[level]:
                if order.size_remaining == 0:
                    self.offers[level].remove(order)
                    orders_purged += 1
                else:
                    orders_remaining += 1
            if len(self.offers[level]) == 0:
                self.offers.remove(level)

        for level in self.bids:
            for order in self.bids[level]:
                if order.size_remaining == 0:
                    self.bids[level].remove(order)
                    orders_purged += 1
                else:
                    orders_remaining += 1
            if len(self.bids[level]) == 0:
                self.bids.remove(level)

        logging.debug('Purged {} orders from the orderbook; leaves {}'.format(orders_purged, orders_remaining))

    def get_bid(self):
        '''Return the best bid on the marketplace'''
        bid_levels = [bid_px for bid_px in self.bids if len(self.bids[bid_px]) > 0]
        try:
            return np.max(bid_levels)
        except:
            return np.nan

    def get_offer(self):
        '''Return the best offer on the marketplace'''
        offer_levels = [ask_px for ask_px in self.offers if len(self.offers[ask_px]) > 0]
        try:
            return np.min(offer_levels)
        except:
            return np.nan
        
    def get_depth(self, price=None, side=None, cumulative=False):

        if (price is None) and (side is None):
            raise Exception('Cannot provide depth of book without a side or a price (side={}, price={})'.format(side, price))
        elif price is None:
            price = self.get_bid() if side=='B' else self.get_ask() if side=='S' else None
        elif side is None:
            side = 'B' if price<=self.get_bid() else 'S' if price>=self.get_offer() else None

        if (price>self.get_bid()) and (price<self.get_offer()):
            logging.info('Depth is zero when price {:,.2f} is between bid {:,.2f} and offer {:,.2f}'.format(price, self.get_bid(), self.get_offer()))
            return 0

        if side=='B':
            if len(self.bids)==0:
                return np.nan
            if cumulative:
                if len([px for px in self.bids if px>=price])==0:
                    return np.nan
                return sum([sum([order.size_remaining for order in self.bids[bid_px]]) for bid_px in self.bids if bid_px>=price])
            else:
                if price not in self.bids:
                    return np.nan
                else:
                    return sum([order.size_remaining for order in self.bids[price]])
        elif side=='S':
            if len(self.offers)==0:
                return np.nan
            if cumulative:
                if len([px for px in self.offers if px<=price])==0:
                    return np.nan
                return sum([sum([order.size_remaining for order in self.offers[ask_px]]) for ask_px in self.ofer if ask_px<=price])
            else:
                if price not in self.offers:
                    return np.nan
                else:
                    return sum([order.size_remaining for order in self.offers[price]])
        else:
            return np.nan

    def get_far(self, order):
        '''Get the far price for a given order'''
        if order.side == 'B':
            return self.get_offer()
        else:
            return self.get_bid()

    def _check_marketable(self, order):
        '''Check if an order is marketable'''
        if (order.side == 'B') and (order.limit >= self.get_offer()):
            order.marketable = True
        elif (order.side == 'S') and (order.limit <= self.get_bid()):
            order.marketable = True
        return order.marketable

    def get_mid(self):
        ''' Return the midprice'''
        return 0.5 * (self.get_bid() + self.get_offer())

    def get_level_2_book(self):
        '''Return the entire limit order book'''
        bids = pd.Series(data=[np.sum([o.size for o in self.bids[p]]) for p in self.bids],
                         index=self.bids, name='bids').to_frame().T

        offers = pd.Series(data=[np.sum([o.size for o in self.offers[p]]) for p in self.offers],
                           index=self.offers, name='offers').to_frame().T

        lvls = pd.concat([bids, offers])
        lvls.fillna(0, inplace=True)

        return lvls

    def send(self, order):
        '''Add an order to the limit order book'''

        if order.size_remaining < 0:
            logging.info('Size remaining {} must be positive'.format(order.size_remaining))
            return []

        # if not math.isclose(order.limit%self.tick_sz, 0):
        #     raise Exception('Order price {} must be a multiple of tick size {}'.format(order.limit, self.tick_sz))

        # Check marketability
        if order.marketable:
            order.limit = self.get_far(order)

        trades = []
        if self._check_marketable(order):
            trades.extend(self._take(order))

        if order.size_remaining > 0:
            # By default, post any residual
            logging.info('Order to {} {} @ {} not fully filled; posting {} residual'.format(order.side, order.size, order.limit, order.size_remaining))
            trades.extend(self._add(order))

        order.prints.extend(trades)

        self.asset.midpoint = self.get_mid() # Update the asset's ex ante fair value

        return trades

    def _init_bid_level(self, price):
        if price not in self.bids:
            self.bids[price] = []

    def _init_offer_level(self, price):
        if price not in self.offers:
            self.offers[price] = []

    def _add(self, order):
        '''Add this order to the limit order book'''
        if order.side == 'B':
            self._init_bid_level(order.limit)
            self.bids[order.limit].append(order)
            logging.debug('Added order for {} to bid level {:,.2f} (size at level now {})'.format(order.size, order.limit, self.get_depth(order.limit)))
        elif order.side == 'S':
            self._init_offer_level(order.limit)
            self.offers[order.limit].append(order)
            logging.debug('Added order for {} to offer level {:,.2f} (size at level now {})'.format(order.size, order.limit, self.get_depth(order.limit)))
        else:
            raise Exception('Not a valid adding order')
        return []

    def _match(self, aggressing_order, resting_orders):
        '''Match an aggressing order against several resting orders'''
        trades = []
        for resting_order in resting_orders:
            fill_size = max(0,min(aggressing_order.size_remaining, resting_order.size_remaining))
            fill_price = resting_order.limit

            if fill_size <= 0:
                logging.info('Fill at ${:,.2f} is not possible ({} demanded against {})'.format(fill_price, aggressing_order.size_remaining, resting_order.size_remaining))
                continue

            if (fill_price < aggressing_order.limit and aggressing_order.size=='S') or (fill_price > aggressing_order.limit and aggressing_order.size=='B'):
                logging.info('Aggressing {} order would fill {} shares at worse price than limit (px={:,.2f}, limit={:,.2f})'.format(aggressing_order.side, fill_size, fill_price, aggressing_order.limit))

            trade = self._print(aggressing_order=aggressing_order, resting_order=resting_order, price=fill_price, size=fill_size)
            trades.append(trade)

            logging.info('{} shares traded at ${:,.2f} ({} initiator; {} agg shs remaining; {} pass shs remaining)'.format(fill_size, fill_price, aggressing_order.side, aggressing_order.size_remaining, self.get_depth(fill_price)))

            if aggressing_order.size_remaining <= 0:
                # Order is fully filled
                break
            
        return trades

    # Print a trade
    def _print(self, aggressing_order, resting_order, price, size):
        '''Create a trade between a resting and aggressive order and print it to the tape'''
        if size <= 0:
            raise Exception('Cannot print a trade with zero quantity')

        aggressing_order.give_fill(price=price, size=size)
        resting_order.give_fill(price=price, size=size)

        self.prints.append({'price': price, 'size': size})

        # If this resting order should be taken off the book
        if resting_order.size_remaining == 0:
            logging.debug('Resting order is fully filled; removing from book')
            if resting_order.side == 'B':
                self.bids[resting_order.limit].remove(resting_order)
            elif resting_order.side == 'S':
                self.offers[resting_order.limit].remove(resting_order)

        # logging.debug('{}@{} between {} and {}'.format(size, price, aggressing_order, resting_order))
        return [price, size]

    def _take(self, order):
        ''' An aggressing order removes liquidity from the book'''
        contra_orders = []
        if order.side == 'B':
            # Find the prices this order could trade at
            far_prices = [p for p in self.offers if p <= order.limit]
            for l in sorted(far_prices):
                contra_orders.extend(self.offers[l])
        elif order.side == 'S':
            # Find the prices this order could trade at
            far_prices = [p for p in self.bids if p >= order.limit]
            for l in sorted(far_prices)[::-1]:
                contra_orders.extend(self.bids[l])

        # Find the orders this order would have to be matched against
        shares_would_fill = 0

        for contra in contra_orders:
            shares_would_fill += contra.size_remaining
            contra_orders.append(contra)
            if shares_would_fill >= order.size:
                break

        # Send these orders to be matched
        trades = self._match(aggressing_order=order, resting_orders=contra_orders)

        if order.size_remaining > 0:
            logging.debug('Aggressing {} order still has {} shares remaining at {:,.2f}'.format(order.side, order.size_remaining, order.limit))

        return trades

    def round_to_tick(self, price, side=None):
        if side=='B':
            price = int(np.floor(price/self.tick_sz)//1)*self.tick_sz
        elif side=='S':
            price = int(np.ceil(price/self.tick_sz)//1)*self.tick_sz
        else:
            price = int(round(price/self.tick_sz)//1)*self.tick_sz
        return round(price,2)