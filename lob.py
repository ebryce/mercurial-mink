import numpy as np
import pandas as pd
import math


class Market:
    def __init__(self, randomize_environment=False):
        self.lob = LOB(
        self.agents=[SeekerAgent(), ProviderAgent()]
        self._randomize_book(n_orders=15)

    def evolve(self):
        for agent in self.agents:
            order=agent.evolve(self)
            if order is not None:
                self.lob.send(order)

    def create_an_order(self, agent):
        new_order=agent.create_an_order()

    def _randomize_book(self, mid=10, spread=0.05, bid_skew=0.5, n_orders=10, depth=5):
        for i in range(n_orders):
            order=self._random_order(
                mid=mid, spread=spread, bid_skew=bid_skew,
                depth=depth, marketable=False)
            self.lob.send(order)

    def _random_order(self, mid=10, side=None, price=None, size=None, spread=0.05,
                      bid_skew=0.5, marketable_skew=0.5, depth=5, marketable=None):

        if side is None:
            side='B' if np.random.rand() < bid_skew else 'S'

        if size is None:
            size=np.random.randint(1, depth) * self.lob.lot_sz

        if price is None:
            if marketable is None:
                marketable=np.random.rand() < marketable_skew

            price_offset_direction=(-1 if side == 'B' else 1) *
                (-1 if marketable else 1)

            price_offset=np.random.rand() * price_offset_direction
            price=round((mid + price_offset) * 100) / 100

        order=Order(side=side, size=size, price=price, marketable=marketable)
        return order


class Agent:
    def __init__(self):
        self.orders=[]
        self.balance=0
        return None

    def work_an_order(self, order):
        self.orders.append(order)
        if order.size == 'B':
            self.balance += order.size
        else:
            self.balance -= order.size
        return order

    def generate_an_order(self):
        return None

    def evolve(self):
        return None

    def bearish(self, size=100, price=None):
        order=Order(side='S', size=size, price=price)
        return order

    def bullish(self, size=100, price=None):
        order=Order(side='B', size=size, price=price)
        return order


class SeekerAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        return None

    def evolve(self, market):
        if market.lob.get_mid() > 10.20:
            return self.bearish(size=1000, price=market.lob.get_bid())
        elif market.lob.get_mid() < 9.80:
            return self.bullish(size=1000, price=market.lob.get_offer())
        return None

    def thesis(self):
        order=Order(side=side, size=size, price=price)
        return order


class ProviderAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        return None

    def evolve(self, market):
        if abs(self.balance) < 10000:
            side='S' if market.lob.get_mid() > 10 else 'B'
            price=market.lob.get_bid() if side == 'B' else market.lob.get_offer()
            if market.lob.get_mid() > 10.20:
                side='S'
                price=10.20
            elif market.lob.get_mid() < 9.80:
                side='B'
                price=9.80

            order=self.risk_on(side=side, price=price)
        else:
            order=self.risk_off()

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

        side='B' if self.balance < 0 else 'S'
        size=(self.balance / 10)

        order=Order(side=side, size=size, marketable=True)
        return order

    def generate_an_order(self):
        return None


class RetailAgent(Agent):
    # Produces nice, un-correlated order flow
    def __init__(self, agent):
        Agent.__init__(self, agent)

    def generate_an_order(self):
        return None


class Order:
    def __init__(self, side, size, price=None, marketable=False):
        self.side=side
        self.size=size
        self.limit=price
        self.marketable=marketable
        self.size_filled=0
        self.size_remaining=self.size - self.size_filled
        self.notional_traded=0
        return None

    def give_fill(self, size, price):
        self.notional_traded += size * price
        self.size_filled += size
        self.size_remaining=self.size - self.size_filled


class LOB:
    def __init__(self):
        self.offers={}
        self.bids={}
        self.prints=[]

        self.tick_sz=0.01
        self.lot_sz=100

        return None

    def get_bid(self):
        bid_levels=[o for o in self.bids if len(self.bids[o]) > 0]
        try:
            return np.min(bid_levels)
        except:
            return 0

    def get_offer(self):
        offer_levels=[o for o in self.offers if len(self.offers[o]) > 0]
        try:
            return np.max(offer_levels)
        except:
            return 999

    def get_far(self, order):
        if order.side == 'B':
            return self.get_offer()
        else:
            return self.get_bid()

    def _check_marketable(self, order):
        if (order.side == 'B') and (order.limit >= self.get_offer()):
            order.marketable=True
        elif (order.side == 'S') and (order.limit <= self.get_bid()):
            order.marketable=True
        return order.marketable

    def get_mid(self):
        return 0.5 * (self.get_bid() + self.get_offer())

    def get_level_2_book(self):

        bids=pd.Series(data=[np.sum([o.size for o in self.bids[p]]) for p in self.bids],
                         index=self.bids, name='bids').to_frame().T

        offers=pd.Series(data=[np.sum([o.size for o in self.offers[p]]) for p in self.offers],
                           index=self.offers, name='offers').to_frame().T

        lvls=pd.concat([bids, offers])
        lvls.fillna(0, inplace=True)

        return lvls

    def send(self, order):
        # Check marketability
        if order.marketable == True:
            order.limit=self.get_far(order)

        if self._check_marketable(order):
            trade_prints=self._take(order)

        if order.size_remaining > 0:
            # By default, post any residual
            trade_prints=self._add(order)

        return trade_prints

    def _init_bid_level(self, price):
        if price not in self.bids:
            self.bids[price]=[]

    def _init_offer_level(self, price):
        if price not in self.offers:
            self.offers[price]=[]

    # Add this order to the limit order book
    def _add(self, order):
        if order.side == 'B':
            self._init_bid_level(order.limit)
            self.bids[order.limit].append(order)
        elif order.side == 'S':
            self._init_offer_level(order.limit)
            self.offers[order.limit].append(order)
        else:
            raise Exception('Not a valid adding order')
        # print('Added {}{}@{:.2f}'.format(order.side, order.size, order.price))
        return []

    # Match an aggressing order against several resting orders
    def _match(self, aggressing_order, resting_orders):
        trade_prints=[]
        for resting_order in resting_orders:
            fill_size=min(aggressing_order.size_remaining,
                            resting_order.size_remaining)
            fill_price=resting_order.limit

            if fill_size <= 0:
                return None

            trade_print=self._print(aggressing_order=aggressing_order,
                                      resting_order=resting_order,
                                      price=fill_price, size=fill_size)
            trade_prints.append(trade_print)

            if aggressing_order.size_remaining <= 0:
                # Order is fully filled
                break

        return trade_prints

    # Print a trade
    def _print(self, aggressing_order, resting_order, price, size):
        if size <= 0:
            return None

        aggressing_order.give_fill(price=price, size=size)
        resting_order.give_fill(price=price, size=size)

        self.prints.append({'price': price, 'size': size})
        # print('Removed {}@{:.2f} by {}'.format(
        #    size, price, aggressing_order.side))

        # If this resting order should be taken off the book
        if resting_order.size_remaining <= 0:
            if resting_order.side == 'B':
                self.bids[resting_order.limit].remove(resting_order)
            elif resting_order.side == 'S':
                self.offers[resting_order.limit].remove(resting_order)

        return [price, size]

    # An agressing order
    def _take(self, order):
        contra_orders=[]
        if order.side == 'B':
            # Find the prices this order could trade at
            far_prices=[p for p in self.offers if p <= order.limit]
            for l in sorted(far_prices):
                contra_orders.extend(self.offers[l])
        elif order.side == 'S':
            # Find the prices this order could trade at
            far_prices=[p for p in self.bids if p >= order.limit]
            for l in sorted(far_prices):
                contra_orders.extend(self.bids[l])

        # Find the orders this order would have to be matched against
        shares_would_fill=0

        for contra in contra_orders:
            shares_would_fill += contra.size_remaining
            contra_orders.append(contra)
            if shares_would_fill >= order.size:
                break

        # Send these orders to be matched
        trade_prints=self._match(aggressing_order=order,
                                   resting_orders=contra_orders)

        return trade_prints
