import numpy as np
import pandas as pd
import math
import logging

logging.basicConfig(level=logging.DEBUG)

class Market:
    def __init__(self, randomize_environment=False):
        self.lob = LOB()
        self.agents = [ProviderAgent()]

        if randomize_environment:
            self._randomize_book(n_orders=15)

    def evolve(self):
        '''Evolves the market by one time step by having each agent submit an order and then evolving the order book'''
        for agent in self.agents:
            order = agent.evolve(self)
            if order is not None:
                trades = self.lob.send(order)
            agent._purge()
        return trades

    def _randomize_book(self, fair_value=10, bid_skew=0.5, n_orders=10, max_depth=5, min_spread=0.05):
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
        
        fair_value = self.lob.get_mid() if fair_value is None else fair_value

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


class Agent:
    def __init__(self):
        self.orders = []
        self.balance = 0
        self.risk_threshold = 10000
        return None

    def _purge(self, aggressive=False):
        for order in self.orders:
            if (order.size_remaining == 0) or aggressive:
                self.orders.remove(order)

    def book_an_order(self, order, market):
        self.orders.append(order)
        if order.size == 'B':
            self.balance += order.size
        else:
            self.balance -= order.size
        return order

    def evolve(self, market):
        if self.balance < self.risk_threshold:
            order = self.generate_an_order(market=market)
            self.book_an_order(order=order, market=market)
        else:
            self._purge(aggressive=True)
        return order

    def bearish(self, size=100, price=None):
        order = Order(side='S', size=size, price=price)
        return order

    def bullish(self, size=100, price=None):
        order = Order(side='B', size=size, price=price)
        return order


class TakerAgent(Agent):
    '''A trader who generally takes liquidity and is willing to pay the spread; they may or may not be informed'''

    def __init__(self, information=0.5):
        '''Information- the probability that the agent is informed'''
        Agent.__init__(self)
        return None

    def generate_an_order(self, market):
        if market.lob.get_mid() > 10.20:
            return self.bearish(size=1000, price=market.lob.get_bid())
        elif market.lob.get_mid() < 9.80:
            return self.bullish(size=1000, price=market.lob.get_offer())
        return None


class ProviderAgent(Agent):
    '''A market maker who posts bids and offers with the intent of earning a spread'''
    def __init__(self):
        Agent.__init__(self)
        return None

    def generate_an_order(self, market):
        if abs(self.balance) < 10000:
            side = 'S' if market.lob.get_mid() > 10 else 'B'
            price = market.lob.get_bid() if side == 'B' else market.lob.get_offer()
            if market.lob.get_mid() > 10.20:
                side = 'S'
                price = 10.20
            elif market.lob.get_mid() < 9.80:
                side = 'B'
                price = 9.80

            order = self.risk_on(side=side, price=price)
        else:
            order = self.risk_off()

        return order

    def risk_on(self, side=None, size=1000, price=None):
        if side == 'B':
            return self.bullish(size=size, price=price)
        else:
            return self.bearish(size=size, price=price)

        return None

    def risk_off(self, urgency=1):
        if self.balance == 0:
            return None

        side = 'B' if self.balance < 0 else 'S'
        size = (self.balance / 10)

        order = Order(side=side, size=size, marketable=True)
        return order

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
        self.notional_traded += size * price
        self.size_filled += size
        self.size_remaining = self.size - self.size_filled


class LOB:
    def __init__(self):
        '''Initiaize an empty limit order book'''
        self.offers = {}
        self.bids = {}
        self.prints = []

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
            logging.warn('Depth is zero when price {:,.2f} is between bid {:,.2f} and offer {:,.2f}'.format(price, self.get_bid(), self.get_offer()))
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
            fill_size = min(aggressing_order.size_remaining, resting_order.size_remaining)
            fill_price = resting_order.limit

            if fill_size <= 0:
                logging.error('Fill at ${:,.2f} is not possible ({} demanded against {})'.format(fill_price, aggressing_order.size_remaining, resting_order.size_remaining))
                continue

            if (fill_price < aggressing_order.limit and aggressing_order.size=='S') or (fill_price > aggressing_order.limit and aggressing_order.size=='B'):
                logging.error('Aggressing {} order would fill {} shares at worse price than limit (px={:,.2f}, limit={:,.2f})'.format(aggressing_order.side, fill_size, fill_price, aggressing_order.limit))

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