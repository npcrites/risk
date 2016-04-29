import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import uuid
import definitions

class Board (object):
    def __init__(self, df, players):
        self.df      = df
        self.players = players

    @classmethod
    def create(cls, players, game_id=None):
        """ Create a new board. 
        
            Arguments:
                players: either (int) number of players to create 
                            or (list) a list of player objects
                game_id: (str) the game identifier
                
            Returns:
                A prepared Board. """
        game_id = str(uuid.uuid1()) if game_id is None else game_id
        players = cls._create_players(players)
        return cls(cls._create_initial_df(game_id, len(players)),
                   players)
        
    @classmethod
    def _create_initial_df(cls, game_id, n_players):
        """ Create DataFrame for an initial board. The regions
            are distributed randomly over the players, and each
            region gets one army as a start. """
        n_rows = 42 + n_players
        return pd.DataFrame({
            'game_id'    : [game_id]*n_rows,
            'turn'       : [-1]*n_rows,
            'event_type' : ['region']*42 + ['card']*n_players,
            'player_id'  : cls._create_random_allocation(n_players) + range(n_players),
            'region_id'  : range(42) + [-1]*n_players,
            'armies'     : [1]*42 + [-1]*n_players,
            'inf_card'   : [0]*n_rows,
            'cav_card'   : [0]*n_rows,
            'can_card'   : [0]*n_rows
        })
        
    @staticmethod
    def _create_players(players):
        """ Create the players, if needed, and match the player_ids. """
        if type(players) == int:
            players = [RandomPlayer() for x in range(players)]
        for i, player in enumerate(players):
            player.player_id = i
        assert len(players) > 1, 'Board: a minimum of two players is required.'
        assert len(players) < 7, 'Board: no more than 6 players can compete.'
        return players
    
    @staticmethod
    def _create_random_allocation(n_players):
        """ Randomly allocate the 42 regions to the players. """
        x = (range(n_players)*42)[0:42]
        random.shuffle(x)
        return x

    # ========== #
    # DF methods #
    # ========== #
    
    def _select_cards(self, player_id=None):
        cards = self.df[self.df.event_type == 'card']
        if player_id is None:
            return cards
        return cards[cards.player_id == player_id]
        
    def _select_regions(self, player_id=None):
        regions = self.df[self.df.event_type == 'region']
        if player_id is None:
            return regions
        return regions[regions.player_id == player_id] 

    # ============= #
    # Basic methods #
    # ============= #       
    
    def current_player(self):
        """ Return the player id of the current turn. """
        return self.turn() % self.n_players() if self.turn() >= 0 else None
    
    def next_turn(self):
        """ Go to the next turn. """
        self.df['turn'] += 1
    
    def turn(self):
        """ Return the current turn number. """
        return self.df.turn.max()
    
    def get_owner(self, region_id):
        return self.df[self.df['region_id'] == region_id]['player_id'].iloc[0]

    def set_owner(self, region_id, player_id):
        self.df.set_value(self.df[self.df.region_id == region_id].index.values, 'player_id', player_id)
    
    def get_armies(self, region_id):
        return self.df[self.df['region_id'] == region_id]['armies'].iloc[0]
    
    def set_armies(self, region_id, n):
        self.df.set_value(self.df[self.df.region_id == region_id].index.values, 'armies', n)
    
    def add_army(self, region_id, n=1):
        # TODO: check current player
        self.set_armies(region_id, self.get_armies(region_id)+n)
        
    def attack_region(self, from_region, to_region, n_armies):
        # TODO: check current player
        # TODO: check n_armies
        attackers = n_armies
        defenders = self.get_armies(to_region)
        att_wins, def_wins = self.fight(attackers, defenders)


        if self.get_armies(to_region) == att_wins:
            self.set_armies(from_region, self.get_armies(from_region)-n_armies)
            self.set_armies(to_region, n_armies - def_wins)
            self.set_owner(to_region, self.get_owner(from_region))
        else:
            self.set_armies(from_region, self.get_armies(from_region)-def_wins)
            self.set_armies(to_region, self.get_armies(to_region)-att_wins)
        
        
    
    # ======== #
    # Pre-game #
    # ======== #    
    
    def init_armies(self):
        """ Have all players place all starting armies on the board. """
        while self.init_single_army():
            continue
        self.next_turn()
            
    def init_single_army(self):
        """ Have each player place one army on the board, if they have one left. """
        changed = False
        for player in self.players:
            if self.n_armies(player.player_id) < self.starting_armies():
                region = player.init(self)
                self.set_armies(region, self.get_armies(region)+1)
                changed = True
        return changed
    
    # ========== #
    # Turn steps #
    # ========== #
    
    def place(self):
        player_id = self.current_player()
        n_reinforcements = self.reinforcements(player_id)
        regions = self.player(player_id).place(self, n_reinforcements)
        for region in regions:
            self.add_army(region)
        # TODO: redeem cards
        # if player.whatdoyouwanttoredeem:
        #     regions = self.pla...
        #        add army
        
        
    def attack(self):
        player_id = self.current_player()
        while True:
            attack = self.player(player_id).attack(self)
            if attack is None:
                return
            from_region, to_region, armies = attack
            self.attack_region(from_region, to_region, armies)
        
    def fortify(self):
        pass
    
    # ======== #
    # Topology #
    # ======== #
    
    @staticmethod
    def continent(continent_id):
        """ Return all region_ids inside the continent. """
        return definitions.continent_regions[continent_id]
    
    def is_internal(self, region_id):
        """ Return True if the region only neighbors regions of the same owner. """
        neighbors = self.neighboring_players(region_id)
        if len(neighbors) > 1: return False
        return neighbors[0] == self.df[self.df.region_id == region_id].player_id.values[0]
    
    @staticmethod
    def neighbors(region_id):
        """ Return the region_ids of the neighbors. """
        return definitions.region_neighbors[region_id]
    
    
    def hostile_neighbors(self, region_id):
        """ Return the region_ids of neighboring regions which are owned by another player. """
        return self.df[self.df.region_id.isin(self.neighbors(region_id)) &
                       (self.df.player_id != self.owner(region_id))].region_id.unique()
    
    def neighboring_players(self, region_id):
        """ Return all players which own a region neighboring this region. """
        return self.df[self.df.region_id.isin(self.neighbors(region_id))].player_id.unique()
            
    
    # ======= #
    # General #
    # ======= #    
    
    def possible_attacks(self, player_id):
        """ Return a DataFrame listing all possible attacks of the player. """
        return self.neighbor_df[(self.neighbor_df.player_id == player_id) &
                                (self.neighbor_df.armies > 1) &
                                (self.neighbor_df.player_id_neighbor != player_id)]
    
    @staticmethod
    def color(player_id):
        return definitions.player_colors[player_id]
    
    @staticmethod
    def location(region_id):
        return definitions.region_locations[region_id]

    def reinforcements(self, player_id):
        base_reinforcements = max(3, self.n_regions(player_id) / 3)
        return base_reinforcements
    
    def n_armies(self, player_id=None):
        return self._select_regions(player_id=player_id).armies.sum()
         
    def n_players(self):
        return self.df['player_id'].nunique()
    
    def n_regions(self, player_id):
        return len(self._select_regions(player_id))
    
    def starting_armies(self):
        return definitions.starting_armies[self.n_players()]
    
    def regions_of(self, pid=None):
        """ Return an array of all regions owner by player pid. """
        return self.df[(self.df.player_id == pid) &
                       (self.df.event_type == 'region')].region_id.values

    def player(self, player_id):
        return self.players[player_id]
    
    # ========== #
    # DataFrames #
    # ========== # 
    
    @property
    def region_df(self):
        """ Return a DataFrame with only the region info. """
        return self.df[self.df.event_type == 'region'][['region_id', 'player_id', 'armies']]
    
    @property
    def neighbor_df(self):
        """ Return a DataFrame with region info of each region merged with its neighbors. """
        return self.region_df.merge(
                   definitions.region_neighbors_df, on='region_id'
               ).merge(
                   self.region_df, left_on='neighbor_id', right_on='region_id', suffixes=('', '_neighbor')
               ).drop('neighbor_id', axis=1)
 
    # ======== #
    # Fighting #
    # ======== #
    
    @classmethod
    def fight(cls, attackers, defenders):
        n_attack_dices = min(attackers, 3)
        n_defend_dices = min(defenders, 2)
        attack_dices = sorted([cls.throw_dice() for i in range(n_attack_dices)], reverse=True)
        defend_dices = sorted([cls.throw_dice() for i in range(n_defend_dices)], reverse=True)
        wins = [att_d > def_d for att_d, def_d in zip(attack_dices, defend_dices)]
        return len([w for w in wins if w is True]), len([w for w in wins if w is False])
        
    @staticmethod
    def throw_dice():
        return random.randint(1,6)
                       
    # ======== #
    # Plotting #
    # ======== #
    
    def plot(self, stats=True, turn=True):
        im = plt.imread(os.getcwd() + '/risk.png')
        plt.figure(figsize=(10, 15))
        implot = plt.imshow(im)
        for i, r in self._select_regions().iterrows():
            self.plot_army(r)          
        if turn: self.plot_turn()
        if stats: self.plot_stats()
        plt.show() 
       
    @classmethod
    def plot_army(cls, row):
        coordinates = cls.location(row.region_id)
        color       = cls.color(row.player_id)
        s = 350
        if row.armies > 9:
            s = 700
        if row.armies > 99:
            s = 1050
        plt.scatter([coordinates[0]], [coordinates[1]], s=s, c=color)
        plt.text(coordinates[0], coordinates[1], str(int(row.armies)),
                 fontsize=20, horizontalalignment='center', verticalalignment='center',
                 color='white' if color not in ['yellow', 'pink'] else 'black')
        
    def plot_stats(self):
        text = ['Plyr: (reg) (arm)']
        for player_id in sorted(self.df.player_id.unique()):
            text.append('{n} {c}: {r} {a}'.format(
                    n=player_id,
                    c=self.color(player_id),
                    r=self.n_regions(player_id),
                    a=self.n_armies(player_id)))
                
        plt.text(50, 1000, '\n'.join(text))
        
    def plot_turn(self):
        turn_text = self.turn() if self.turn() >= 0 else 'pre-game'
        plt.text(50, 1100, 'Turn: {t} - {c}'.format(t=turn_text,
                                                    c=self.color(self.current_player())),
                 fontsize=15)
    
    @property
    def state(self):
        return self.df.drop('game_id', axis=1)
    
class Player (object):
    
    def __init__(self, name=None):
        self.name      = name
        self.player_id = None

    def attack(self, b):
        raise Exception('base player cannot play')
    def fortify(self, b):
        raise Exception('base player cannot play')
    def place(self, b):
        raise Exception('base player cannot play')        

class RandomPlayer (Player):
    
    def attack(self, board):
        if random.random() > 0.75:
            return None
        possible_attacks = board.possible_attacks(self.player_id)
        if len(possible_attacks) == 0:
            return None
        attack = possible_attacks.sample()
        print attack['region_id'].iloc[0], attack['region_id_neighbor'].iloc[0], attack['armies'].iloc[0]-1
        return attack['region_id'].iloc[0], attack['region_id_neighbor'].iloc[0], attack['armies'].iloc[0]-1
    def fortify(self, board):
        pass
    def init(self, board):
        return random.choice(board.regions_of(self.player_id))
    def place(self, board, n):
        return [self.init(board) for x in range(n)]

        
        
    
    
    