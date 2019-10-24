import tensorflow as tf
import numpy as np
import collections
import os, sys
import random

board = None
player_cnt = None
player_turn = None

layer_sizes = [64, 100, 200, 400, 200, 64]
std_dev = 0.01

learning_rate = 0.00001
epsilon = 1.0

match_per_iter = 100
max_iter = 10

e_downrate = 0.9 / max_iter

gamma = 0.9

epochs = 50

win_reward = 10
loss_reward = -3
draw_reward = 1


# Angka untuk menandakan disk yang ada pada board
# 0 = kosong
# 1 = item
# 2 = putih

class Coord:
    def __init__(self, _y, _x=None):
        if _x != None:
            self.y = _y
            self.x = _x
        elif _x == None:
            num=_y
            self.y = num//8
            self.x = num%8
    def __add__(self, a):
        return Coord(self.y+a.y, self.x+a.x)
    def __repr__(self):
        return ('Coord(%d, %d)'%(self.y, self.x))
    def __eq__(self, a):
        return self.x==a.x and self.y==a.y
    def __ne__(self, a):
        return self.x!=a.x or self.y!=a.y
    def to_num(self):
        return self.y*8+self.x

directions = {
    'l' : Coord( 0, -1), # kiri
    'ul': Coord(-1, -1), # kiri atas
    'u' : Coord(-1,  0), # atas
    'ur': Coord(-1,  1), # atas kanan
    'r' : Coord( 0,  1), # kanan
    'dr': Coord( 1,  1), # kanan bawah
    'd' : Coord( 1,  0), # bawah
    'dl': Coord( 1, -1) # kanan kiri
}

def is_valid_coord(coord):
    # returns true if coord is in board, false otherwise
    return 0<=coord.x<=7 and 0<=coord.y<=7

def other_color(p_color):
    # returns other player's color code
    return [0, 2, 1][p_color]

# OTHELLO FUNCTIONSS
def actions(coord, p_color=None):
    # returns
    #     [] if invalid move
    #     list of directions and terminal coordinates
    #     [(direction, terminal coordinates)] if valid
    if not is_valid_coord(coord): return []
    if board[coord.y, coord.x] != 0: return []

    if not p_color: p_color = player_turn

    v_move = []

    for direction in directions.values():
        # adj  = adjacent
        # term = terminate
        adj = coord + direction
        
        if is_valid_coord(adj) and board[adj.y, adj.x] == other_color(p_color):
            while is_valid_coord(adj) and board[adj.y, adj.x] == other_color(p_color):
                adj = adj + direction

            if is_valid_coord(adj) and board[adj.y, adj.x] == p_color:
                v_move.append((direction, adj))
            
    return v_move

def all_valid_moves(p_color=None):
    if not p_color: p_color=player_turn
    valid_moves = []
    for y in range(8):
        for x in range(8):
            mv = actions(Coord(y, x), p_color)
            if mv:
                valid_moves.append(Coord(y, x))
    return valid_moves

def make_move(coord, p_color=None):
    global player_turn
    if not p_color: p_color = player_turn

    moves = actions(coord, p_color)
    if not moves: return False

    board[coord.y, coord.x] = p_color
    player_cnt[p_color] += 1
    player_cnt[0] -= 1

    for dr, mv in moves:
        tmp_coord = coord+dr

        while tmp_coord != mv:
            board[tmp_coord.y, tmp_coord.x] = p_color
            player_cnt[p_color] += 1
            player_cnt[other_color(p_color)] -= 1

            tmp_coord = tmp_coord+dr
    
    player_turn = other_color(player_turn)

def get_game_state():
    # returns
    #     (x, y)
    #     x =
    #        -1 if not finished
    #        0 if board filled
    #        1 if a color is gone
    #        2 if both can't move
    #     y = delta of color count
    #         positive if black wins, negative otherwise
    delta = player_cnt[1] - player_cnt[2]
    if player_cnt[1]+player_cnt[2] == 64: return (0, delta)
    if player_cnt[1] == 0: return (1, delta)
    if player_cnt[2] == 0: return (1, delta)
    if not all_valid_moves(1) and not all_valid_moves(2): return (2, delta)
    return (-1, delta)

def init_game():
    global board
    global player_cnt
    global player_turn

    board = np.array([[0]*8]*8)
    player_cnt = [60, 2, 2]

    board[3, 3] = board[4, 4] = 2
    board[4, 3] = board[3, 4] = 1

    player_turn = 1

def _test_game_state():
    global board
    global player_cnt
    
    test_results=[]
    expected_results = [
        ( 1,  6),
        ( 1, -6),
        (-1, -4),
        ( 0, 26),
        ( 0, -6),
    ]
    # kemakan abis, item menang
    board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 1, 1, 1, 0, 0],[0, 0, 0, 1, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])
    player_cnt = [b-1 for a,b in sorted([(a,b) for a,b in collections.Counter(board.flatten().tolist()+[0, 1, 2]).items()])]
    test_results.append(get_game_state())
    
    # kemakan abis, putih menang
    board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 2, 2, 2, 0, 0],[0, 0, 0, 2, 2, 2, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])
    player_cnt = [b-1 for a,b in sorted([(a,b) for a,b in collections.Counter(board.flatten().tolist()+[0, 1, 2]).items()])]
    test_results.append(get_game_state())

    # blm slese
    board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 2, 2, 2, 0, 0],[0, 0, 0, 1, 2, 2, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])
    player_cnt = [b-1 for a,b in sorted([(a,b) for a,b in collections.Counter(board.flatten().tolist()+[0, 1, 2]).items()])]
    test_results.append(get_game_state())
    
    # penuh, hitam menang
    board = np.array([[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[2, 2, 2, 2, 2, 2, 2, 2],[2, 2, 2, 2, 2, 2, 2, 2],[1, 1, 1, 2, 2, 2, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])
    player_cnt = [b-1 for a,b in sorted([(a,b) for a,b in collections.Counter(board.flatten().tolist()+[0, 1, 2]).items()])]
    test_results.append(get_game_state())

    # penuh, putih menang
    board = np.array([[2, 2, 2, 2, 2, 2, 2, 2],[2, 2, 2, 2, 2, 2, 2, 2],[2, 2, 2, 2, 2, 2, 2, 2],[2, 2, 2, 2, 2, 2, 2, 2],[1, 1, 1, 2, 2, 2, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]])
    player_cnt = [b-1 for a,b in sorted([(a,b) for a,b in collections.Counter(board.flatten().tolist()+[0, 1, 2]).items()])]
    test_results.append(get_game_state())
    

    name = ['NOT FINISHED', 'FULL', 'WON']
    failed = False
    for i, (res, expected) in enumerate(zip(test_results, expected_results)):
        if res != expected:
            print('TEST FAILED: GAME #%d %s %s found. should be %s %s'%(i, name[res[0]+1], res[1], name[expected[0]+1], expected[1]))
            failed = True
    if not failed: print('TEST PASSED: game_state()')

def _test_valid_moves():
    expected_valid = [
        [(Coord(0, 1), Coord(3, 4))],
        [(Coord(1, 0), Coord(4, 3))],
        [(Coord(1, 1), Coord(4, 4))]
    ]

    init_game()
    invalid = []
    invalid.append(actions(Coord(3, 3), 1)) # uda keisi
    invalid.append(actions(Coord(8, 5), 1)) # diluar board
    make_move(Coord(2, 3), 1)
    invalid.append(actions(Coord(3, 2), 2)) # ga ada ujung

    init_game()
    valid = []
    valid.append(actions(Coord(3, 2), 1)) # di kiri
    valid.append(actions(Coord(2, 3), 1)) # di atas
    make_move(Coord(2, 3), 1)
    valid.append(actions(Coord(2, 2), 2)) # di kiri atas
    
    failed = False
    for i, x in enumerate(invalid):
        if x:
            print('TEST FAILED: INVALID MOVE #%d DETECTED AS %s'%(i, str(x)))
            failed = True
    for i, (x,y) in enumerate(zip(valid, expected_valid)):
        if x!=y:
            print('TEST FAILED: VALID MOVE #%d %s DETECTED AS %s'%(i, str(y), str(x)))
            failed = True
    if not failed: print('TEST PASSED: valid_moves()')

def _test_make_move():
    #p_cnt, dan isi dari board yang di-expect
    expected_valid = [
        ([57, 5, 2], np.array([[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 2, 1, 0, 0, 0, 0],[0, 0, 1, 1, 1, 0, 0, 0],[0, 0, 0, 1, 2, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])),
        ([56, 3, 5], np.array([[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 2, 0, 0, 0, 0, 0],[0, 0, 2, 2, 1, 0, 0, 0],[0, 0, 2, 1, 2, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])),
        ([52, 5, 7], np.array([[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 2, 2, 2, 2, 2],[0, 0, 0, 2, 1, 0, 0, 0],[0, 0, 2, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])),
        ([54, 5, 5], np.array([[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 1, 0],[0, 0, 2, 2, 2, 2, 2, 0],[0, 0, 0, 1, 1, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])),
        ([52, 3, 9], np.array([[0, 0, 0, 0, 0, 0, 2, 0],[0, 0, 0, 2, 0, 2, 0, 0],[0, 0, 1, 2, 2, 0, 0, 0],[0, 0, 0, 2, 2, 0, 0, 0],[0, 0, 2, 2, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]]))
    ]
    
    #bisa (test pass)
    # 1=item, 2=putih
    init_game()
    valid = []
    moves = [(Coord(2, 3), 1), (Coord(2, 2), 2), (Coord(3, 2), 1)]
    for mv in moves: make_move(mv[0], mv[1])
    valid.append((player_cnt, np.copy(board)))

    init_game()
    moves = [(Coord(3, 2), 1), (Coord(4, 2), 2), (Coord(5, 3), 1), (Coord(2, 2), 2)]
    for mv in moves: make_move(mv[0], mv[1])
    valid.append((player_cnt, np.copy(board)))

    init_game()
    moves = [(Coord(5, 4), 1), (Coord(3, 5), 2), (Coord(2, 5), 1), (Coord(1, 5), 2), (Coord(3, 6), 1), (Coord(5, 2), 2), (Coord(0, 5), 1), (Coord(3, 7), 2)]
    for mv in moves: make_move(mv[0], mv[1])
    valid.append((player_cnt, np.copy(board)))
    
    init_game()
    moves = [(Coord(4,5),1),(Coord(3,5),2),(Coord(2,6),1),(Coord(3,6),2),(Coord(2,3),1),(Coord(3,2),2)]
    for mv in moves: make_move(mv[0], mv[1])
    valid.append((player_cnt, np.copy(board)))

    init_game()
    moves = [(Coord(2,3),1), (Coord(2,4),2), (Coord(1,5),1), (Coord(4,2),2), (Coord(5,5),1), (Coord(0,6),2), (Coord(2,2),1), (Coord(1,3),2)]
    for mv in moves: make_move(mv[0], mv[1])
    valid.append((player_cnt, np.copy(board)))
    
    failed = False
    for i, (x,y) in enumerate(zip(valid,expected_valid)):
        state_failed = x[0]!=y[0]
        board_failed = (x[1]!=y[1]).any()
        if state_failed:
            print('TEST FAILED : MAKE MOVE #%d p_cnt %s expected, %s found'%(i, str(y[0]), str(x[0])))
            failed = True
        if board_failed:
            print('TEST FAILED : MAKE MOVE #%d different board expected'%(i))
            failed = True
    if not failed: print('TEST PASSED: make_move()')

def _test_all():
    _test_valid_moves()
    _test_make_move()
    _test_game_state()

def print_board():
    print(' ', ' '.join([str(x) for x in range(8)]))
    for y in range(8):
        print(y, '', end='')
        for x in range(8):
            if actions(Coord(y, x)): print('* ', end='')
            else: print(['. ', 'x ', 'o '][board[y, x]], end='')
        print(y, '')
    print(' ', ' '.join([str(x) for x in range(8)]))    

# ANN FUNCTIONS
def make_network():
    weights = [
        tf.Variable(tf.random.truncated_normal([a,b], stddev=std_dev))
        for a,b in zip(layer_sizes, layer_sizes[1:])
    ]

    biases = [
        tf.Variable(tf.random.normal([x]))
        for x in layer_sizes[1:]
    ]

    layers = [tf.placeholder("float", [None, layer_sizes[0]])]
    for w,b in zip(weights, biases):
        layers.append(tf.nn.relu(tf.matmul(layers[-1], w) + b))

    prediction = tf.argmax(layers[-1][0])

    target_Qout = tf.placeholder("float", [None, layer_sizes[-1]])
    loss =  tf.reduce_mean(tf.square(tf.subtract(target_Qout, layers[-1])))

    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return layers[0], layers[-1], prediction, target_Qout, loss, train

def save_network(sess): 
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
    input()

def load_network(sess):
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored")

def train_network():
    global epsilon
    global match_per_iter
    global max_iter
    global e_downrate
    global gamma
    global epochs

    X, Qout, pred, target_Qout, loss, train = make_network()
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    try:
        load_network(sess)
        input('Model loaded. Press enter')
    except:
        print("Can't find old model")

    while True:
        loss_sum = 0
        loss_ctr = 0
        matches = [play_game(sess, X, pred) for _ in range(match_per_iter)]
        for i in range(epochs):
            random.shuffle(matches)
            for match in matches:
                print(match[-1][-1])
                print(match[-1][-2])
                for j, (cur, cur_valid, move, reward, next_state) in enumerate(match):
                    if j != 0:
                        next_Q = sess.run(Qout, feed_dict={X: [next_state]})
                        max_next_Q = np.max(next_Q)
                        reward = gamma * max_next_Q

                    target_Q = sess.run(Qout, feed_dict={X: [cur]})

                    for i, _ in enumerate(cur):
                        if Coord(i) not in cur_valid:
                            target_Q[0, i] = -1
                    
                    target_Q[0, move.to_num()] = reward

                    _, _, t_loss = sess.run([train, Qout, loss], feed_dict={X: [cur], target_Qout: target_Q})
                    loss_sum += t_loss
                    loss_ctr += 1
                print(loss_sum/loss_ctr)
            loss_sum /= loss_ctr
            loss_ctr = 1
            save_network(sess)

        print('\n'*10, loss_sum/loss_ctr, '\n'*10)
        if epsilon > 0: epsilon -= e_downrate
        else: epsilon = random.random()/300

def ann_make_move(sess, X, pred):
    pred_move = sess.run(pred, feed_dict={X: [board.flatten()]})

    valid_moves = all_valid_moves()
    
    move = Coord(pred_move)
    if not actions(move) or random.random() < epsilon:
        move = random.choice(valid_moves)

    return move

def opp_make_move(sess, X, pred):
    board_cpy = np.copy(board)
    for y in range(8):
        for x in range(8):
            board_cpy[y, x] = other_color(board_cpy[y, x])
    
    pred_move = sess.run(pred, feed_dict={X: [board_cpy.flatten()]})

    valid_moves = all_valid_moves()
    
    move = Coord(pred_move)
    if not actions(move) or random.random() < 0.2:
        move = random.choice(valid_moves)

    return move

def play_game(sess, X, pred):
    global board
    global player_turn

    init_game()

    complete_memory = []

    while True:
        while True:
            ann_move = ann_make_move(sess, X, pred)

            turn_memory = [np.copy(board), all_valid_moves(), ann_move]

            make_move(ann_move)
            
            game_state = get_game_state()
            
            if game_state[0] != -1:
                turn_memory += [
                    loss_reward if game_state[1] < 0
                    else draw_reward if game_state[1] == 0
                    else win_reward,
                    np.copy(board)
                ]
                complete_memory.append(turn_memory)
                break
                
            # print_board()

            if not all_valid_moves():
                player_turn = other_color(player_turn)
            else: break
            
        while True:
            game_state = get_game_state()
            if game_state[0] != -1: break
            
            opp_move = opp_make_move(sess, X, pred)
            # while True:
            #     opp_move = list(map(int, input('input [y x]: ').split()))
            #     opp_move = Coord(opp_move[0], opp_move[1])
            #     if actions(opp_move): break

            make_move(opp_move)

            game_state = get_game_state()
            
            if game_state[0] != -1:
                turn_memory += [
                    loss_reward if game_state[1] < 0
                    else draw_reward if game_state[1] == 0
                    else win_reward,
                    np.copy(board)
                ]
                complete_memory.append(turn_memory)
                break

            # print_board()

            if not all_valid_moves():
                player_turn = other_color(player_turn)
            else: break
        
        game_state = get_game_state()
        if game_state[0] == -1:
            turn_memory += [(game_state[0]-game_state[1])//10, np.copy(board)]
            complete_memory.append(turn_memory)
        if game_state[0] != -1: break
    
    print_board()
    print(get_game_state())

    for m in complete_memory:
        m[0] = m[0].flatten()
        m[-1] = m[-1].flatten()

    return complete_memory


def main():
    train_network()

debug = True
if __name__ == '__main__':
    if debug: _test_all()
    main()
    
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . . . . . . 1
# 2 . . . . . . . . 2
# 3 . . . o x * . . 3
# 4 . . . x x . . . 4
# 5 . . . * x * . . 5
# 6 . . . . . . . . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 5 5
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . . . . . . 1
# 2 . . . * . . . . 2
# 3 . . * o x . . . 3
# 4 . . . x o * . . 4
# 5 . . . . x o * . 5
# 6 . . . . . . . . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 11520 thread 4 bound to OS proc set 1
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 16704 thread 5 bound to OS proc set 3
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 12140 thread 6 bound to OS proc set 5
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 16896 thread 7 bound to OS proc set 7
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . . . . . . 1
# 2 . . * x * . . . 2
# 3 . . . x x . . . 3
# 4 . . * x o . . . 4
# 5 . . . * x o . . 5
# 6 . . . . * . . . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 2 2
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . . . . . . 1
# 2 . * o x . . . . 2
# 3 . . * o x . . . 3
# 4 . . . x o * . . 4
# 5 . . . . x o * . 5
# 6 . . . . . . . . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 16624 thread 8 bound to OS proc set 0
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 7624 thread 9 bound to OS proc set 2
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 8168 thread 10 bound to OS proc set 4
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 2820 thread 11 bound to OS proc set 6
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . * . . . . 1
# 2 . . o x * . . . 2
# 3 . . . o x * . . 3
# 4 . . . x x x . . 4
# 5 . . . * x o . . 5
# 6 . . . . . . . . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 5 3
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . * . . . . . . 1
# 2 . * o x . . . . 2
# 3 . . * o x . . . 3
# 4 . . * o x x . . 4
# 5 . . * o o o . . 5
# 6 . . * * * * * . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . * . * . . . . 1
# 2 . x x x * * . . 2
# 3 . . . o x * * . 3
# 4 . . . o x x * . 4
# 5 . . . o o o . . 5
# 6 . . . . . . . . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 2 4
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . . . . . . 1
# 2 . x x x o * . . 2
# 3 . . . o o . . . 3
# 4 . . * o o x . . 4
# 5 . . . o o o . . 5
# 6 . . . * . * * . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 6348 thread 12 bound to OS proc set 1
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 7836 thread 13 bound to OS proc set 3
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 2808 thread 14 bound to OS proc set 5
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 6204 thread 15 bound to OS proc set 7
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . * * . . . . . 1
# 2 * x x x o . . . 2
# 3 . . * x o * . . 3
# 4 . . * x o x * . 4
# 5 . . * x x o * . 5
# 6 . . * x * . . . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 5 2
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . . . * . . 1
# 2 . x x x o * . . 2
# 3 . . . x o * . . 3
# 4 . * * o o x . . 4
# 5 . . o o o o . . 5
# 6 . . . x . * * . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . * * * * x . . 1
# 2 . x x x x . . . 2
# 3 . . * x o * * . 3
# 4 . . . o o x * . 4
# 5 . . o o o o * . 5
# 6 . . . x . . . . 6
# 7 . . * * * . . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 3 2
#   0 1 2 3 4 5 6 7
# 0 . . . . . . . . 0
# 1 . . . . . x . . 1
# 2 . x x x x . . . 2
# 3 . . o o o . . . 3
# 4 . * * o o x . . 4
# 5 . . o o o o . . 5
# 6 . . . x * * * . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 16864 thread 16 bound to OS proc
# set 0
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 948 thread 17 bound to OS proc set 2
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 5332 thread 18 bound to OS proc set 4
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 10440 thread 19 bound to OS proc
# set 6
#   0 1 2 3 4 5 6 7
# 0 . . . . . . * . 0
# 1 * * * * * x . . 1
# 2 . x x x x * . . 2
# 3 . . o o x * . . 3
# 4 . . . o x x * . 4
# 5 . . o o x o . . 5
# 6 . . . x x * . . 6
# 7 . . . * * * . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 0 6
#   0 1 2 3 4 5 6 7
# 0 . . . . . . o . 0
# 1 . . . . * o . . 1
# 2 . x x x o * . . 2
# 3 . * o o x . . . 3
# 4 . * * o x x * . 4
# 5 . * o o x o * . 5
# 6 . * * x x * * . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 12316 thread 20 bound to OS proc
# set 1
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 16460 thread 21 bound to OS proc
# set 3
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 13804 thread 22 bound to OS proc
# set 5
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 16748 thread 23 bound to OS proc
# set 7
#   0 1 2 3 4 5 6 7
# 0 . . . . . * o . 0
# 1 * * * * x o . . 1
# 2 . x x x x * . . 2
# 3 . . o o x * . . 3
# 4 . . . o x x * . 4
# 5 . . o o x o . . 5
# 6 . . . x x * . . 6
# 7 . . . * * * . . 7
#   0 1 2 3 4 5 6 7
# input [y x]: 0 5
#   0 1 2 3 4 5 6 7
# 0 . . . . * o o . 0
# 1 . . * * o o . . 1
# 2 . x x o x . . . 2
# 3 . * o o x . . . 3
# 4 . * * o x x * . 4
# 5 . * o o x o * . 5
# 6 . * * x x * * . 6
# 7 . . . . . . . . 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 . . . . . o o . 0
# 1 * * * . o o . . 1
# 2 * x x o x * . . 2
# 3 . . o o x * . . 3
# 4 . . . o x x * . 4
# 5 . . o o x x * . 5
# 6 . . . x x * x . 6
# 7 . . . * * * . * 7
#   0 1 2 3 4 5 6 7
# input [y x]: 7 7
#   0 1 2 3 4 5 6 7
# 0 . . . . * o o . 0
# 1 . . * * o o . . 1
# 2 . x x o x . . . 2
# 3 . * o o x . . . 3
# 4 . * * o o x * . 4
# 5 . * o o x o * . 5
# 6 . * . x x * o . 6
# 7 . . . . . . . o 7
#   0 1 2 3 4 5 6 7
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 10060 thread 24 bound to OS proc
# set 0
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 5204 thread 25 bound to OS proc set 2
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 17260 thread 26 bound to OS proc
# set 4
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 1896 thread 27 bound to OS proc set 6
#   0 1 2 3 4 5 6 7
# 0 . . . . . o o . 0
# 1 * * . . o o . . 1
# 2 * x x o x * . . 2
# 3 . x x x x * . . 3
# 4 . * * o o x * . 4
# 5 . . o o x o * . 5
# 6 . . . x x * o . 6
# 7 . . . * * * . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 2 0
#   0 1 2 3 4 5 6 7
# 0 . . . . * o o . 0
# 1 * * * * o o . . 1
# 2 o o o o x . . . 2
# 3 . x x x x . . . 3
# 4 . * * o o x * . 4
# 5 . * o o x o * . 5
# 6 . * . x x * o . 6
# 7 . . . . . . . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 * . * . . o o . 0
# 1 * x . . o o . . 1
# 2 o x x o x * . . 2
# 3 . x x x x * . . 3
# 4 . * * o o x * . 4
# 5 . . o o x o * . 5
# 6 . . . x x * o . 6
# 7 . . . * * * . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 0 0
#   0 1 2 3 4 5 6 7
# 0 o * . . * o o . 0
# 1 . o * * o o . . 1
# 2 o x o o x . . . 2
# 3 . x x o x . . . 3
# 4 . * * o o x * . 4
# 5 . * o o x o * . 5
# 6 . * . x x * o . 6
# 7 . . . . . . . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o . . . . o o . 0
# 1 . o . . o o . . 1
# 2 o x o o x * . . 2
# 3 . x x x x * . . 3
# 4 * * x x x x . . 4
# 5 . * o x x o * . 5
# 6 . . . x x . o . 6
# 7 . . . * * * . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 7 4
#   0 1 2 3 4 5 6 7
# 0 o * . . . o o . 0
# 1 . o * * o o . . 1
# 2 o x o o o * . . 2
# 3 . x x x o * . . 3
# 4 . . x x o x . . 4
# 5 . * o x o o * . 5
# 6 . * * o o * o . 6
# 7 . . * * o * . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o . . . . o o . 0
# 1 . o . . o o . . 1
# 2 o x o o o . . . 2
# 3 * x x x o . * . 3
# 4 * * x x o x * . 4
# 5 . * o x x x x * 5
# 6 . . * o o . o * 6
# 7 . . . . o . . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 5 7
#   0 1 2 3 4 5 6 7
# 0 o * . . . o o . 0
# 1 . o * * o o . . 1
# 2 o x o o o * . . 2
# 3 . x x x o * . . 3
# 4 . . x x o x . . 4
# 5 . . o o o o o o 5
# 6 . * * o o * o * 6
# 7 . . * * o * . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o . . . . o o . 0
# 1 * o . . o o . . 1
# 2 o x o o o . . . 2
# 3 * x x x o * * . 3
# 4 * * x x o x * . 4
# 5 . * o x o o o o 5
# 6 . . * x o . o . 6
# 7 . . * x o . . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 1 0
#   0 1 2 3 4 5 6 7
# 0 o * . . . o o . 0
# 1 o o * * o o . . 1
# 2 o o o o o . . . 2
# 3 . x o x o * . . 3
# 4 . * x o o x * . 4
# 5 . * o x o o o o 5
# 6 . . * x o * o * 6
# 7 . . . x o * . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o * * * . o o . 0
# 1 o o x * o o . . 1
# 2 o o x x o * . . 2
# 3 . x x x x * * . 3
# 4 . * x o o x * . 4
# 5 . * o x o o o o 5
# 6 . . * x o . o . 6
# 7 . . * x o . . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 0 1
#   0 1 2 3 4 5 6 7
# 0 o o * . . o o . 0
# 1 o o o * o o . . 1
# 2 o o x o o * . . 2
# 3 . x x x o * * . 3
# 4 . * x o o o * . 4
# 5 . * o x o o o o 5
# 6 . . * x o * o . 6
# 7 . . . x o * . o 7
#   0 1 2 3 4 5 6 7
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 9860 thread 28 bound to OS proc set 1
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 9388 thread 29 bound to OS proc set 3
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 14596 thread 30 bound to OS proc
# set 5
# OMP: Info #250: KMP_AFFINITY: pid 1444 tid 7560 thread 31 bound to OS proc set 7
#   0 1 2 3 4 5 6 7
# 0 o o x * . o o . 0
# 1 o o x * o o . . 1
# 2 o o x o o . . . 2
# 3 * x x x o . . . 3
# 4 . * x o o o . . 4
# 5 . * o x o o o o 5
# 6 . . * x o . o . 6
# 7 . . * x o . . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 0 3
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o * o o . . 1
# 2 o o x o o * . . 2
# 3 . x x x o * * . 3
# 4 . * x o o o * . 4
# 5 . * o x o o o o 5
# 6 . . * x o * o . 6
# 7 . . . x o * . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o . o o . . 1
# 2 o o x o o . . . 2
# 3 * x x x o . . . 3
# 4 * x x o o o . . 4
# 5 * * x x o o o o 5
# 6 . * * x o . o . 6
# 7 . . * x o . . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 3 0
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o * o o . . 1
# 2 o o x o o * . . 2
# 3 o o o o o * . . 3
# 4 * o x o o o * . 4
# 5 . * o x o o o o 5
# 6 . . * o o . o . 6
# 7 . . . x o * . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o * o o . . 1
# 2 o o x o o . . . 2
# 3 o o o o o * * . 3
# 4 . o x x x x x * 4
# 5 . * o x o x o o 5
# 6 . . . o x * o . 6
# 7 . . * x o * . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 1 3
#   0 1 2 3 4 5 6 7
# 0 o o o o * o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o * . . 2
# 3 o o o o o . . . 3
# 4 * o x x x x x . 4
# 5 . * o x o x o o 5
# 6 . * * o x * o * 6
# 7 . . * x o * * o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o . . . 2
# 3 o o o o o * * . 3
# 4 . o x x x x x * 4
# 5 . * x x o x o o 5
# 6 . x * o x * o . 6
# 7 * . * x o * . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 3 7
# input [y x]: 4 7
#   0 1 2 3 4 5 6 7
# 0 o o o o * o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o * . . 2
# 3 o o o o o * . * 3
# 4 . o o o o o o o 4
# 5 . . x x o x o o 5
# 6 . x * o x . o . 6
# 7 . . . x o * . o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o . . * 2
# 3 o o o o o * * x 3
# 4 . o o o o o x o 4
# 5 . * x x o x o o 5
# 6 . x * o x * o . 6
# 7 * . * x o * . o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 2 7
#   0 1 2 3 4 5 6 7
# 0 o o o o * o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o * . o 2
# 3 o o o o o * . o 3
# 4 * o o o o o x o 4
# 5 . . x x o x o o 5
# 6 . x * o x . o . 6
# 7 . . . x o * * o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o . . o 2
# 3 o o o o o * . o 3
# 4 . o o o o o x o 4
# 5 . * x x o x x o 5
# 6 . x * o x * x * 6
# 7 * . * x o * x o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 7 5
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o * . o 2
# 3 o o o o o * . o 3
# 4 * o o o o o x o 4
# 5 . . x o o x x o 5
# 6 . x . o o . o . 6
# 7 . . . x o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o x * o 2
# 3 o o o o x * * o 3
# 4 . o o x o o x o 4
# 5 . * x o o x x o 5
# 6 . x * o o * o * 6
# 7 . . * x o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 3 5
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o * . 1
# 2 o o o o o o . o 2
# 3 o o o o o o * o 3
# 4 * o o x o o o o 4
# 5 . . x o o x x o 5
# 6 . x . o o * o . 6
# 7 . . . x o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o o . o 2
# 3 o o o o o o . o 3
# 4 . o o x o o o o 4
# 5 . * x o x x x o 5
# 6 . x * o o x o * 6
# 7 * . * x o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o * o o . 0
# 1 o o o o o o * . 1
# 2 o o o o o o . o 2
# 3 o o o o o o * o 3
# 4 * o o x o o o o 4
# 5 . . x o x x o o 5
# 6 . x * o o x o o 6
# 7 . . * x o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o o * o 2
# 3 o o o o o o x o 3
# 4 . o o x o x o o 4
# 5 . * x o x x o o 5
# 6 . x * o o x o o 6
# 7 * . * x o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 2 6
#   0 1 2 3 4 5 6 7
# 0 o o o o * o o . 0
# 1 o o o o o o * . 1
# 2 o o o o o o o o 2
# 3 o o o o o o o o 3
# 4 * o o x o x o o 4
# 5 . . x o x x o o 5
# 6 . x * o o x o o 6
# 7 . . * x o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o o o o 2
# 3 o o o o o o o o 3
# 4 x x x x o x o o 4
# 5 * * x o x x o o 5
# 6 . x * o o x o o 6
# 7 * . * x o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 7 0
#   0 1 2 3 4 5 6 7
# 0 o o o o * o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o o o o 2
# 3 o o o o o o o o 3
# 4 x x x o o x o o 4
# 5 . * o o x x o o 5
# 6 . o * o o x o o 6
# 7 o . * x o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o . o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o o o o 2
# 3 o o o o o o o o 3
# 4 x x x o o x o o 4
# 5 * * o o x x o o 5
# 6 . o * x o x o o 6
# 7 o * x x o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 7 1
#   0 1 2 3 4 5 6 7
# 0 o o o o * o o . 0
# 1 o o o o o o . . 1
# 2 o o o o o o o o 2
# 3 o o o o o o o o 3
# 4 x x x o o x o o 4
# 5 . * o o x x o o 5
# 6 . o * x o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o x o o . 0
# 1 o o o x x o . . 1
# 2 o o x o x o o o 2
# 3 o x o o x o o o 3
# 4 x x x o x x o o 4
# 5 * * o o x x o o 5
# 6 . o * x o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 6 2
#   0 1 2 3 4 5 6 7
# 0 o o o o x o o * 0
# 1 o o o x x o * * 1
# 2 o o x o x o o o 2
# 3 o x o o x o o o 3
# 4 x x x o x x o o 4
# 5 . * o o x x o o 5
# 6 * o o o o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o x o o . 0
# 1 o o o x x o . . 1
# 2 o o x o x o o o 2
# 3 o x o o x o o o 3
# 4 x x x o x x o o 4
# 5 * x x x x x o o 5
# 6 * o o o o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 6 0
#   0 1 2 3 4 5 6 7
# 0 o o o o x o o * 0
# 1 o o o x x o * * 1
# 2 o o x o x o o o 2
# 3 o x o o x o o o 3
# 4 x x o o x x o o 4
# 5 * o x x x x o o 5
# 6 o o o o o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o x x x x 0
# 1 o o o x x o . . 1
# 2 o o x o x o o o 2
# 3 o x o o x o o o 3
# 4 x x o o x x o o 4
# 5 * o x x x x o o 5
# 6 o o o o o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 5 0
#   0 1 2 3 4 5 6 7
# 0 o o o o x x x x 0
# 1 o o o x x o * * 1
# 2 o o x o x o o o 2
# 3 o x o o x o o o 3
# 4 o o o o x x o o 4
# 5 o o x x x x o o 5
# 6 o o o o o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
#   0 1 2 3 4 5 6 7
# 0 o o o o x x x x 0
# 1 o o o x x o * x 1
# 2 o o x o x o x o 2
# 3 o x o o x x o o 3
# 4 o o o o x x o o 4
# 5 o o x x x x o o 5
# 6 o o o o o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
# input [y x]: 1 6
#   0 1 2 3 4 5 6 7
# 0 o o o o x x x x 0
# 1 o o o x x o o x 1
# 2 o o x o x o o o 2
# 3 o x o o x x o o 3
# 4 o o o o x x o o 4
# 5 o o x x x x o o 5
# 6 o o o o o x o o 6
# 7 o o o o o o o o 7
#   0 1 2 3 4 5 6 7
# (0, -26)