import AlphaZero.GI
using StaticArrays
using Crayons

const BOARD_SIZE = 6

if isodd(BOARD_SIZE)
    error("Board size needs to be even!")
end

const Player = UInt8
# the first player has to be white according to AlphaZero.jl
const WHITE = 0x01
const BLACK = 0x02
const EMPTY = 0x00

other(p::Player) = 0x03 - p

const Board = SMatrix{BOARD_SIZE, BOARD_SIZE, Player, BOARD_SIZE^2}

function initial_value(p)
    k = BOARD_SIZE/2
    if p == (k, k) || p == (k+1, k+1)
        return WHITE
    elseif p == (k+1, k) || p == (k, k+1)
        return BLACK
    else
        return EMPTY
    end
end

const INITIAL_BOARD = @SMatrix [initial_value((i, j)) for i in 1:BOARD_SIZE, j in 1:BOARD_SIZE] 

# the first player has to be white according to AlphaZero.jl
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

mutable struct Game <: GI.AbstractGame
    board::Board
    curplayer::Player
    history::Vector{Tuple{Int, Int}}
end

function Game()
    board = INITIAL_STATE.board
    curplayer = INITIAL_STATE.curplayer
    history = []
    Game(board, curplayer, history)
end

GI.State(::Type{Game}) = typeof(INITIAL_STATE)

GI.Action(::Type{Game}) = Tuple{Int, Int}
GI.two_players(::Type{Game}) = true

function GI.game_terminated(g::Game)
    if legal_moves(g) == [(-1, -1)] && legal_moves(Game(g.board, other(g.curplayer), [])) == [(-1,-1)]
        return true
    else
        return false
    end
end
GI.white_playing(g::Game) = g.curplayer == WHITE
GI.white_playing(::Type{Game}, state) = state.curplayer == WHITE

GI.white_reward(g::Game) = if winner(g) == WHITE
    1.
        elseif winner(g) == BLACK
    -1.
        else
    0.
        end

GI.current_state(g::Game) = (board=g.board, curplayer=g.curplayer)


function Game(state)
    return Game(state.board, state.curplayer, [])
end

# (-1, -1) is a passing action
GI.actions(::Type{Game}) = [[(i, j) for i in 1:BOARD_SIZE for j in 1:BOARD_SIZE]; [(-1, -1)]]

function path_length_in_direction(g::Game, start, direction)
    for i in 1:BOARD_SIZE
        idx = start .+ (i.*direction)
        if checkbounds(Bool, g.board, idx...)
            if g.board[idx...] == g.curplayer
                return i-1
            elseif g.board[idx...] == EMPTY
                return 0
            end
        else
            return 0
        end
    end
    return 0
end

function valid_directions(g::Game, start)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),  (1, 1), (-1,- 1), (1, -1), (-1, 1)]
    return filter(d -> path_length_in_direction(g, start, d)>0, directions)
end

function islegal(g::Game, a)
    # if we're not passing
    if a != (-1, -1)
        return g.board[a...] == EMPTY && length(valid_directions(g, a))>0
    else
        non_passing_moves = filter(m -> m != a, GI.actions(Game))
        return length(filter(move -> islegal(g, move), non_passing_moves))==0
    end
end

function legal_moves(g::Game)
    filter(a -> islegal(g, a), GI.actions(Game))
end

function GI.actions_mask(g::Game)
    map(a -> islegal(g, a), GI.actions(Game))
end

function GI.heuristic_value(g::Game)
    return convert(AbstractFloat, length(findall(isequal(g.curplayer), g.board)))
end

function GI.vectorize_state(::Type{Game}, state)
    # we pretend it's WHITE to play
    f(p::Player) = if state.curplayer == WHITE
        return p
    else
        return other(p)
    end

    return Float32[state.board[i, j] == c
                   for i in 1:BOARD_SIZE,
                   j in 1:BOARD_SIZE,
                   c in [EMPTY, f(WHITE), f(BLACK)]]
end

function GI.symmetries(::Type{Game}, state)
    # 90 degree rotation
    σ((x, y)) = if (x,y) == (-1,-1)
        return (-1,-1)
    else
        (BOARD_SIZE + 1, 0) .+ (-y, x)
    end

    # mirror
    τ((x, y)) = if (x,y) == (-1,-1)
        return (-1,-1)
    else
        return (BOARD_SIZE - x + 1, y)
    end

    # D4 \ {id}
    syms = [σ, σ∘σ, σ∘σ∘σ, τ, τ∘σ, τ∘σ∘σ, τ∘σ∘σ∘σ]

    return [
        (
            (
                board = Board([state.board[f((i,j))...] for i in 1:BOARD_SIZE, j in 1:BOARD_SIZE]),
                curplayer = state.curplayer
            ),
            # is this the right understanding of permutation? or do i need the inverse one here?
            map(c -> findfirst(isequal(f(c)), GI.actions(Game)), GI.actions(Game))
        )
        for f in syms
    ]
end

function winner(g::Game)
    if GI.game_terminated(g)
        white_count = length(findall(isequal(WHITE), g.board))
        black_count = length(findall(isequal(BLACK), g.board))
        if white_count > black_count
            return WHITE
        elseif black_count > white_count
            return BLACK
        end
    end
    return nothing
end

function GI.play!(g::Game, action)
    push!(g.history, action)

    if action != (-1, -1)
        for d in valid_directions(g, action)
            for i in 0:path_length_in_direction(g, action, d)
                g.board = setindex(g.board, g.curplayer, (action .+ (i .* d))...)
            end
        end
    end
    g.curplayer = other(g.curplayer)
end


####################
# UI functions
####################

function GI.action_string(::Type{Game}, action)
    if action == (-1,-1)
        return "pass"
    else
        return Char(96+action[2])*string(action[1])
    end
end

function GI.parse_action(g::Game, str)
    if str == "pass"
        return (-1, -1)
    end

    col = Int(str[1])-Int('a')+1
    row = parse(Int, str[2:length(str)])
    return (row, col)
end


function GI.render(g::Game)
    colors = [crayon"light_red", crayon"light_blue"]
    symbols = ["x", "o"]

    # print the first line
    print("  ")
    for i in 1:BOARD_SIZE
        print(Char(96+i))
    end
    print("\n")
    for i in 1:BOARD_SIZE
        print(i, "|")

        for j in 1:BOARD_SIZE
            v = g.board[i, j]
            if v != EMPTY
                print(colors[v], symbols[v], crayon"reset")
            else
                print(".")
            end
        end

        print("|", i)
        print("\n")
    end

    print("  ")
    for i in 1:BOARD_SIZE
        print(Char(96+i))
    end
    print("\n")

    println("Player 1: ", colors[1], symbols[1], crayon"reset")
    println("Player 2: ", colors[2], symbols[2], crayon"reset")
end


