from risk.board import Board,Territory
import copy

# define boards for the tests
# HINT: I recommend you plot these boards
board0 = Board([Territory(territory_id=i, player_id=1, armies=i%5+1) for i in range(42)])

board1 = copy.deepcopy(board0)
board1.set_owner(3,0)
board1.set_armies(3,500)
board1.set_armies(4,500)
board1.set_armies(34,500)
board1.set_armies(39,500)
board1.set_armies(35,500)

board2 = copy.deepcopy(board1)
board2.set_owner(34, 0)
board2.set_armies(34,1)
board2.set_owner(35, 0)
board2.set_armies(35,1)
board2.set_owner(39, 0)
board2.set_armies(39,1)

board3 = copy.deepcopy(board2)
board3.set_owner(34, 1)

board4 = copy.deepcopy(board2)
board4.set_owner(4, 0)
board4.set_armies(4, 1)
board4.set_owner(1, 0)

board5 = Board([Territory(territory_id=i, player_id=i%5, armies=i%5+1) for i in range(42)])


def test__is_valid_path_1():
    assert board5.is_valid_path([])

def test__is_valid_path_2():
    assert board5.is_valid_path([6])

def test__is_valid_path_3():
    assert not board5.is_valid_path([6,34])

def test__is_valid_path_4():
    assert not board5.is_valid_path([6,31,6])

def test__is_valid_path_5():
    assert board5.is_valid_path([5,37,4,24,39,34,35,36,0,6,31])

def test__is_valid_path_6():
    assert not board5.is_valid_path([5,37,4,34,35,36,0,6,31])
    


def test__is_valid_attack_path_1():
    assert not board1.is_valid_attack_path([])

def test__is_valid_attack_path_2():
    assert not board1.is_valid_attack_path([5])

def test__is_valid_attack_path_3():
    assert not board1.is_valid_attack_path([5,6])

def test__is_valid_attack_path_4():
    assert board1.is_valid_attack_path([3,28])

def test__is_valid_attack_path_5():
    assert board1.is_valid_attack_path([3, 28, 37, 5, 40, 2, 1, 19, 22, 6, 15, 21, 11, 24])

def test__is_valid_attack_path_6():
    assert board5.is_valid_attack_path([3, 4, 24, 11, 21, 0, 6])

def test__is_valid_attack_path_6():
    print("here")
    print(board5.is_valid_attack_path([3, 4, 24, 8, 21, 0, 6]))
    assert not board5.is_valid_attack_path([3, 4, 24, 8, 21, 0, 6])

def test__is_valid_attack_path_7():
    assert not board5.is_valid_attack_path([3, 4, 24, 11, 21, 6])



def test__cost_of_attack_path_1():
    assert board1.cost_of_attack_path([3,28]) == 4

def test__cost_of_attack_path_2():
    assert board1.cost_of_attack_path([3, 28, 37, 5, 40, 2, 1, 19, 22, 6, 15, 21, 11, 24]) == 34

def test__cost_of_attack_path_3():
    assert board5.cost_of_attack_path([3, 4, 24, 11, 21, 0, 6]) == 17


def test__shortest_path_1():
    assert len(board1.shortest_path(3,3)) == 1

def test__shortest_path_2():
    assert len(board1.shortest_path(0,13)) == 5

def test__shortest_path_3():
    assert len(board5.shortest_path(31,23)) == 3

def test__shortest_path_4():
    assert len(board5.shortest_path(3,32)) == 8

def test__shortest_path_5():
    assert len(board5.shortest_path(1,40)) == 3

def test__shortest_path_6():
    assert len(board2.shortest_path(15,16)) == 3

def test__shortest_path_7():
    assert len(board3.shortest_path(15,16)) == 3

def test__shortest_path_8():
    assert len(board1.shortest_path(34,41)) == 5

def test__shortest_path_9():
    assert len(board1.shortest_path(1,0)) == 5

def test__shortest_path_10():
    assert len(board1.shortest_path(14,38)) == 8


def test__can_fortify_1():
    assert not board1.can_fortify(3,10)

def test__can_fortify_2():
    assert not board2.can_fortify(3,39)

def test__can_fortify_3():
    assert board2.can_fortify(39,35)

def test__can_fortify_4():
    assert board2.can_fortify(25,38)

def test__can_fortify_5():
    assert not board5.can_fortify(25,38)


def test__cheapest_attack_path_1():
    assert board1.cost_of_attack_path(board1.cheapest_attack_path(3,24)) == 34

def test__cheapest_attack_path_2():
    assert board2.cost_of_attack_path(board2.cheapest_attack_path(3,24)) == 34

def test__cheapest_attack_path_3():
    assert board3.cost_of_attack_path(board3.cheapest_attack_path(3,24)) == 31

def test__cheapest_attack_path_4():
    assert board4.cheapest_attack_path(3,24) is None

def test__cheapest_attack_path_5():
    assert board5.cost_of_attack_path(board5.cheapest_attack_path(3,24)) == 10

def test__cheapest_attack_path_6():
    assert board5.cost_of_attack_path(board5.cheapest_attack_path(30,24)) == 13

def test__cheapest_attack_path_7():
    assert board5.cost_of_attack_path(board5.cheapest_attack_path(3,6)) == 17

def test__cheapest_attack_path_8():
    assert board5.cheapest_attack_path(3,18) is None

def test__cheapest_attack_path_9():
    assert board5.cost_of_attack_path(board5.cheapest_attack_path(38,1)) == 16

def test__cheapest_attack_path_10():
    assert board5.cost_of_attack_path(board5.cheapest_attack_path(3,2)) == 13

def test__cheapest_attack_path_11():
    assert board5.cheapest_attack_path(3,3) is None


def test__can_attack_1():
    assert board1.can_attack(3,24)

def test__can_attack_2():
    assert board2.can_attack(3,24)

def test__can_attack_3():
    assert board3.can_attack(3,24)

def test__can_attack_4():
    assert not board4.can_attack(3,24)

def test__can_attack_5():
    assert board5.can_attack(3,24)

def test__can_attack_6():
    assert board5.can_attack(30,24)

def test__can_attack_7():
    assert board5.can_attack(3,6)

def test__can_attack_8():
    assert not board5.can_attack(3,18)

def test__can_attack_9():
    assert board5.can_attack(38,1)

def test__can_attack_10():
    assert board5.can_attack(3,2)

def test__can_attack_11():
    assert not board5.can_attack(3,3)
