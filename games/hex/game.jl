import AlphaZero.GI
using StaticArrays
using Crayons

const BOARD_SIZE = 5
const Player = UInt8
# Player 1
const WHITE = 0x01
# Player 2
const BLACK = 0x02
const EMPTY = 0x00

other(p::Player) = 0x03 - p

const Board = SMatrix{BOARD_SIZE, BOARD_SIZE, Player, BOARD_SIZE^2}

const INITIAL_BOARD = @SMatrix zeros(Player, BOARD_SIZE, BOARD_SIZE)
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE, switched=false)


mutable struct Game <: GI.AbstractGame
    board::Board
    curplayer::Player
    switched::Bool
    winner::Player
    history::Vector{Tuple{Int, Int}}
end

function Game()
    board = INITIAL_STATE.board
    curplayer = INITIAL_STATE.curplayer
    switched = INITIAL_STATE.switched
    winner = 0x00
    history = []
    Game(board, curplayer, switched, winner, history)
end

GI.State(::Type{Game}) = typeof(INITIAL_STATE)

# we encode the switch as (-1, -1), so all actions are Tuple{Int, Int}
GI.Action(::Type{Game}) = Tuple{Int, Int}
GI.two_players(::Type{Game}) = true

GI.game_terminated(g::Game) = g.winner != 0x00
GI.white_playing(g::Game) = g.curplayer == WHITE
GI.white_playing(::Type{Game}, state) = state.curplayer == WHITE

GI.white_reward(g::Game) = if g.winner == WHITE
    1.
        elseif g.winner == BLACK
    -1.
        else
    0.
        end

GI.current_state(g::Game) = (board=g.board, curplayer=g.curplayer, switched=g.switched)

function Game(state)
    g = Game()
    g.board = state.board
    g.curplayer = state.curplayer
    g.switched = state.switched
    if winner(g.board) !== nothing
        g.winner = winner(g.board)
    end
    return g
end

GI.actions(::Type{Game}) = [[(i, j) for i in 1:BOARD_SIZE for j in 1:BOARD_SIZE]; [(-1,-1)]]

function islegal(g::Game, a)
    if a != (-1, -1)
        g.board[a...] == EMPTY
    else
        (!GI.white_playing(g)) && length([(i, j) for i in 1:BOARD_SIZE for j in 1:BOARD_SIZE if g.board[i,j] != EMPTY]) == 1
    end
end

function GI.actions_mask(g::Game)
    map(a -> islegal(g, a), GI.actions(Game))
end

# TODO implement me
function GI.heuristic_value(g::Game)
    return 0.
end

function GI.vectorize_state(::Type{Game}, state)
    # FIXME how to implement the switched board? what does it even mean?
    return Float32[state.board[i, j] == c
                   for i in 1:BOARD_SIZE,
                   j in 1:BOARD_SIZE,
                   c in [EMPTY, WHITE, BLACK]]
end

function GI.symmetries(::Type{Game}, state)
    flip((x, y)) = if (x,y) != (-1,-1)
        (BOARD_SIZE - x + 1, BOARD_SIZE - y + 1)
    else
        (-1,-1)
    end

    # TODO is this correct? or do we have an issue with perm vs perm^{-1}? as
    # the symmetry is self-inverse, it shouldn't matter though.
    return [
        ((board = Board([state.board[flip((i,j))...] for i in 1:BOARD_SIZE, j in 1:BOARD_SIZE]),
          curplayer=state.curplayer, switched=state.switched),
         map(c -> findfirst(isequal(c), GI.actions(Game)),
             map(flip, GI.actions(Game))))
    ]
end



#####################
# path finding logic
#####################

function distance(a, b)
    # to have an easier time, we switch them so p is higher on the board than q
    if a[1] < b[1]
        p = a
        q = b
    else
        p = b
        q = a
    end
    r_diff = q[1]-p[1] # this is now \geq 0
    c_diff = q[2]-p[2]

    # we have an advantage over the manhattan distance if the differences have opposite signs
    if r_diff*c_diff < 0
        diag = min(abs(r_diff), abs(c_diff))
        return diag + distance((p[1]+diag, p[2]-diag), q)
    else
        return abs(r_diff) + abs(c_diff)
    end
end

function theoretical_neighbours(p)
    i, j = p[1], p[2]
    return [(i,j), (i-1, j), (i-1,j+1), (i,j-1), (i,j+1), (i+1,j), (i+1, j-1)]
end

function reachable_in_one(occupied_fields, start)
    return filter(x -> in(x, theoretical_neighbours(start)), occupied_fields)
end

function reachable(occupied_fields, start)
    tmp = reachable_in_one(occupied_fields, start)
    next = vcat(map(s -> reachable_in_one(occupied_fields, s), tmp)...)
    while tmp != next
        tmp = next
        next = sort(unique(vcat(map(s -> reachable_in_one(occupied_fields, s), tmp)...)))
    end
    return next
end

function has_path(fields, starts, targets)
    if findfirst(in(fields), starts) === nothing
        return false
    end

    # checking whether a path is theoretically possible gives a very minor performance boost
    distances = [distance(s, t) for s in starts for t in targets]
    if min(distances...) > length(fields)
        return false
    end

    # superfluous nodes are nodes in starts which we have already reached
    superfluous = []
    for s in starts
        if !(s in superfluous) && in(s, fields)
            nodes = reachable(fields, s)
            if findfirst(in(targets), nodes) !== nothing
                return true
            end

            superfluous = vcat(superfluous, filter(in(starts), nodes))
            fields = filter(!in(nodes), fields)
        end
    end
    return false
end

function winner(board::Board)
    white_fields = [(i, j) for i in 1:BOARD_SIZE for j in 1:BOARD_SIZE if board[i, j] == WHITE]
    black_fields = [(i, j) for i in 1:BOARD_SIZE for j in 1:BOARD_SIZE if board[i, j] == BLACK]

    # horizontal path for WHITE, vertical path for BLACK
    if has_path(white_fields, [(i, 1) for i in 1:BOARD_SIZE], [(i, BOARD_SIZE) for i in 1:BOARD_SIZE])
        return WHITE
    elseif has_path(black_fields, [(1, j) for j in 1:BOARD_SIZE], [(BOARD_SIZE, j) for j in 1:BOARD_SIZE])
        return BLACK
    else
        return nothing
    end
end


function GI.play!(g::Game, action)
    push!(g.history, action)

    # are we switching?
    if action == (-1, -1)
        # the first played field is now actually BLACK
        played_moves = [(i, j) for i in 1:BOARD_SIZE for j in 1:BOARD_SIZE if g.board[i, j] != EMPTY]
        g.board = setindex(g.board, BLACK, played_moves[1]...)
        g.switched = true
        g.curplayer = WHITE
        return nothing
    end

    g.board = setindex(g.board, g.curplayer, action...) 
    g.curplayer = other(g.curplayer)

    # is the game over now?
    if !isnothing(winner(g.board))
        g.winner = winner(g.board)
    end
end


####################
# UI functions
####################

function GI.action_string(::Type{Game}, action)
    if action == (-1,-1)
        return "S"
    else
        return Char(96+action[2])*string(action[1])
    end
end

function GI.parse_action(g::Game, str)
    if str == "S" || str == "s"
        return (-1, -1)
    end

    col = Int(str[1])-Int('a')+1
    row = parse(Int, str[2:length(str)])
    return (row, col)
end


function GI.render(g::Game)
    if BOARD_SIZE > 26
        error("not implemented")
    end
    # print the first line
    for i in 1:BOARD_SIZE
        print(" ")
        print(crayon"light_blue", Char(96+i), crayon"reset")
        print(" ")
    end
    print("\n")
    for i in 1:BOARD_SIZE
        leading_spaces = i-1
        if i>9
            leading_spaces -= 1
        end
        for k in 1:leading_spaces
            print(" ")
        end

        print(crayon"light_red", i, crayon"reset")
        print("\\")

        for j in 1:BOARD_SIZE
            v = g.board[i, j]
            if v != EMPTY
                print([crayon"light_red", crayon"light_blue"][v])
                print(["x", "o"][v])
                print(crayon"reset")
            else
                print(".")
            end

            if j<BOARD_SIZE
                print("  ")
            end
        end

        print("\\")
        print(crayon"light_red", i, crayon"reset")
        print("\n")
    end

    # print last line
    leading_spaces = 1+BOARD_SIZE
    for k in 1:leading_spaces
        print(" ")
    end
    for i in 1:BOARD_SIZE
        print(" ")
        print(crayon"light_blue", Char(96+i), crayon"reset")
        print(" ")
    end
    print("\n")

    println("Player 1: ", crayon"light_red", "x", crayon"reset", " horizontally")
    println("Player 2: ", crayon"light_blue", "o", crayon"reset", " vertically")
end


