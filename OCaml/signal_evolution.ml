(* SignalNet Evolution: Multi-Agent Neural Trading Evolution System *)

(* Core data structures *)
type tick = { bid: float; ask: float; last: float; volume: float; timestamp: float }

type activation_fn = Relu | LeakyRelu | Tanh | Swish | Mish | Sigmoid | GELU | ELU | Adaptive

let all_activations = [| Relu; LeakyRelu; Tanh; Swish; Mish; Sigmoid; GELU; ELU; Adaptive |]

(* Helper functions for arrays *)
let index_of x arr =
  let rec aux i =
    if i >= Array.length arr then raise Not_found
    else if arr.(i) = x then i
    else aux (i + 1)
  in aux 0

let max_by_array f arr =
  let max_idx = ref 0 in
  let max_val = ref (f 0 arr.(0)) in
  for i = 1 to Array.length arr - 1 do
    let v = f i arr.(i) in
    if v > !max_val then begin
      max_val := v;
      max_idx := i;
    end
  done;
  !max_idx

let take n lst =
  let rec aux i acc = function
    | [] -> List.rev acc
    | x :: xs -> if i <= 0 then List.rev acc else aux (i - 1) (x :: acc) xs
  in
  aux n [] lst

(* Advanced activation functions *)
let activate fn x =
  match fn with
  | Relu -> max 0.0 x
  | LeakyRelu -> if x > 0.0 then x else 0.01 *. x
  | Tanh -> tanh x
  | Swish -> x /. (1.0 +. exp (-.x))
  | Mish -> x *. tanh (log (1.0 +. exp x))
  | Sigmoid -> 1.0 /. (1.0 +. exp (-.x))
  | GELU -> 0.5 *. x *. (1.0 +. tanh (sqrt (2.0 /. Float.pi) *. (x +. 0.044715 *. x *. x *. x)))
  | ELU -> if x > 0.0 then x else 1.0 *. (exp x -. 1.0)
  | Adaptive -> if x > 0.0 then x *. tanh x else 0.1 *. x

(* Neural network layer *)
type neural_layer = {
  weights: float array array;
  biases: float array;
  activation: activation_fn;
  mutable contribution: float;
}

type neural_network = {
  mutable layers: neural_layer array;
  mutable fitness_history: float list;
}

(* Enhanced genome with runner intelligence *)
type genome = {
  alpha_fast: float;
  alpha_slow: float;
  rsi_threshold: float;
  rsi_period: int;
  take_profit_mult: float;
  stop_loss_mult: float;
  fib_ratio1: float;
  fib_ratio2: float;
  reward_risk_weight: float;
  ode_accel_coeff: float;
  ode_jerk_coeff: float;
  runner_hold_threshold: float;
  runner_exit_threshold: float;
  volatility_sensitivity: float;
  fibonacci_targets: float array;
  neural_depth: int;
  neural_width: int;
  preferred_activation: activation_fn;
  runner_patience: float;
  risk_adaptation: float;
}

(* Dynamic state with runner intelligence *)
type state = {
  mutable ema_fast: float;
  mutable ema_slow: float;
  mutable rsi: float;
  mutable price: float;
  mutable momentum: float;
  mutable acceleration: float;
  mutable jerk: float;
  mutable gain: float;
  mutable loss: float;
  mutable atr: float;
  mutable volume_ema: float;
  mutable signal_strength: float;
  mutable runner_mode: bool;
  mutable runner_score: float;
  mutable consecutive_wins: int;
  mutable position_hold_time: float;
  mutable volatility: float;
  mutable trend_strength: float;
  mutable fib_support: float;
  mutable fib_resistance: float;
}

(* Communication system *)
type signal = float array

type message = {
  sender_id: int;
  signal: signal;
  weight: float;
  trust_score: float;
}

type comm_agent = {
  id: int;
  mutable inbox: message list;
  mutable outbox: message list;
  mutable knowledge_base: (string * float) list;
  mutable trust_scores: (int * float) list;
  mutable memory_decay_rate: float;
  mutable message_filter: signal -> bool;
}

let default_filter _ = true

(* Activation policy for meta-learning *)
type activation_policy = {
  mutable weights: float array;
  mutable history: (activation_fn * float) list;
  mutable performance_map: (activation_fn * float) list;
}

(* Enhanced agent with neural network and communication *)
type agent = {
  id: int;
  genome: genome;
  neural_net: neural_network;
  mutable fitness: float;
  mutable state: state;
  mutable total_pnl: float;
  mutable trade_count: int;
  mutable win_count: int;
  activation_policy: activation_policy;
  comms: comm_agent;
  mutable generation: int;
  mutable runner_trades: (float * float * float) list; (* entry, exit, duration *)
  mutable adaptation_rate: float;
}

(* Initialize neural network *)
let create_neural_network depth width input_size =
  let layers = Array.make depth { 
    weights = Array.make_matrix 1 1 0.0; 
    biases = Array.make 1 0.0; 
    activation = Relu; 
    contribution = 0.0 
  } in
  for i = 0 to depth - 1 do
    let layer_input = if i = 0 then input_size else width in
    let layer_output = width in
    let weights = Array.make_matrix layer_output layer_input (Random.float 0.2 -. 0.1) in
    let biases = Array.make layer_output (Random.float 0.1 -. 0.05) in
    let activation = all_activations.(Random.int (Array.length all_activations)) in
    layers.(i) <- { weights; biases; activation; contribution = 0.0 }
  done;
  { layers; fitness_history = [] }

(* Forward pass through neural network *)
let forward_pass nn inputs =
  Array.fold_left (fun acc layer ->
    let outputs = Array.mapi (fun i bias ->
      let weighted_sum = Array.fold_left2 (fun sum w input -> sum +. w *. input) bias layer.weights.(i) acc in
      let activated = activate layer.activation weighted_sum in
      layer.contribution <- layer.contribution +. abs_float activated;
      activated
    ) layer.biases in
    outputs
  ) inputs nn.layers

(* Communication functions *)
let route_signal sender agents signal =
  List.iter (fun agent ->
    if agent.id <> sender.id && agent.comms.message_filter signal then
      let weight = Random.float 1.0 in
      let trust = try List.assoc sender.id agent.comms.trust_scores with Not_found -> 0.5 in
      let msg = { sender_id = sender.id; signal; weight; trust_score = trust } in
      agent.comms.inbox <- msg :: agent.comms.inbox
  ) agents

let update_trust agent sender_id delta =
  let updated = List.map (fun (id, score) ->
    if id = sender_id then (id, max 0.0 (min 1.0 (score +. delta))) else (id, score)
  ) agent.comms.trust_scores in
  agent.comms.trust_scores <- updated

let decay_knowledge agent =
  agent.comms.knowledge_base <- List.map (fun (k, v) -> 
    (k, v *. (1.0 -. agent.comms.memory_decay_rate))
  ) agent.comms.knowledge_base

let process_inbox agent =
  let score msg =
    let signal_strength = Array.fold_left (+.) 0.0 msg.signal in
    signal_strength *. msg.weight *. msg.trust_score
  in
  let summary = List.fold_left (fun acc msg -> acc +. score msg) 0.0 agent.comms.inbox in
  agent.comms.knowledge_base <- ("signal_strength", summary) :: agent.comms.knowledge_base;
  decay_knowledge agent;
  agent.comms.inbox <- []

(* Activation policy functions *)
let select_activation policy input_signal_quality =
  let score_fn i _ =
    let w = policy.weights.(i) in
    w *. input_signal_quality +. (Random.float 0.05)
  in
  let idx = max_by_array score_fn all_activations in
  let chosen = all_activations.(idx) in
  policy.history <- (chosen, input_signal_quality) :: policy.history;
  chosen

let reward_activation_policy policy reward =
  List.iter (fun (fn, quality) ->
    let idx = index_of fn all_activations in
    policy.weights.(idx) <- policy.weights.(idx) +. (reward *. quality)
  ) policy.history;
  policy.history <- []

(* Initialize state *)
let init_state price = {
  ema_fast = price;
  ema_slow = price;
  rsi = 50.0;
  price = price;
  momentum = 0.0;
  acceleration = 0.0;
  jerk = 0.0;
  gain = 0.0;
  loss = 0.0;
  atr = 0.0;
  volume_ema = 0.0;
  signal_strength = 0.0;
  runner_mode = false;
  runner_score = 0.0;
  consecutive_wins = 0;
  position_hold_time = 0.0;
  volatility = 0.0;
  trend_strength = 0.0;
  fib_support = price *. 0.618;
  fib_resistance = price *. 1.618;
}

(* Fibonacci analysis *)
let calculate_fibonacci_levels high low =
  let range = high -. low in
  let levels = [| 0.0; 0.236; 0.382; 0.5; 0.618; 0.786; 1.0 |] in
  Array.map (fun level -> low +. range *. level) levels

(* Advanced state update with ODE dynamics *)
let update_state state tick genome nn =
  let dt = 1.0 in
  let price_change = tick.last -. state.price in
  
  (* Update EMAs with adaptive smoothing *)
  let alpha_fast = genome.alpha_fast *. (1.0 +. state.volatility *. genome.volatility_sensitivity) in
  let alpha_slow = genome.alpha_slow *. (1.0 +. state.volatility *. genome.volatility_sensitivity) in
  
  state.ema_fast <- state.ema_fast +. alpha_fast *. (tick.last -. state.ema_fast);
  state.ema_slow <- state.ema_slow +. alpha_slow *. (tick.last -. state.ema_slow);
  
  (* Higher-order derivatives *)
  let old_momentum = state.momentum in
  let old_acceleration = state.acceleration in
  
  state.momentum <- price_change /. dt;
  state.acceleration <- (state.momentum -. old_momentum) /. dt;
  state.jerk <- (state.acceleration -. old_acceleration) /. dt;
  
  (* Apply ODE coefficients *)
  state.acceleration <- state.acceleration *. genome.ode_accel_coeff;
  state.jerk <- state.jerk *. genome.ode_jerk_coeff;
  
  (* RSI with exponential smoothing *)
  if price_change > 0.0 then begin
    state.gain <- state.gain *. 0.93 +. price_change *. 0.07;
    state.loss <- state.loss *. 0.93;
  end else begin
    state.gain <- state.gain *. 0.93;
    state.loss <- state.loss *. 0.93 +. (-.price_change) *. 0.07;
  end;
  
  let rs = if state.loss > 0.0 then state.gain /. state.loss else 100.0 in
  state.rsi <- 100.0 -. (100.0 /. (1.0 +. rs));
  
  (* Volatility estimation *)
  state.volatility <- state.volatility *. 0.95 +. (abs_float price_change) *. 0.05;
  
  (* Trend strength *)
  state.trend_strength <- abs_float (state.ema_fast -. state.ema_slow) /. state.price;
  
  (* Fibonacci levels *)
  let high = max tick.last state.price in
  let low = min tick.last state.price in
  let fib_levels = calculate_fibonacci_levels high low in
  state.fib_support <- fib_levels.(2); (* 38.2% level *)
  state.fib_resistance <- fib_levels.(4); (* 61.8% level *)
  
  (* Neural network signal *)
  let nn_input = [| state.ema_fast; state.ema_slow; state.rsi; state.momentum; 
                   state.acceleration; state.jerk; state.volatility; state.trend_strength |] in
  let nn_output = forward_pass nn nn_input in
  state.signal_strength <- if Array.length nn_output > 0 then nn_output.(0) else 0.0;
  
  (* Update price *)
  state.price <- tick.last

(* Runner intelligence *)
let update_runner_mode state genome last_trade_result =
  if last_trade_result > 0.0 then
    state.consecutive_wins <- state.consecutive_wins + 1
  else
    state.consecutive_wins <- 0;
  
  let momentum_factor = abs_float state.momentum in
  let win_factor = float_of_int state.consecutive_wins in
  let trend_factor = state.trend_strength in
  
  state.runner_score <- momentum_factor *. win_factor *. trend_factor *. state.signal_strength;
  
  if state.runner_score > genome.runner_hold_threshold then
    state.runner_mode <- true
  else if state.runner_score < genome.runner_exit_threshold then
    state.runner_mode <- false

(* Signal generation with runner intelligence *)
let generate_signals ticks agent =
  let signals = ref [] in
  let rec process_ticks remaining_ticks last_trade_result =
    match remaining_ticks with
    | [] -> List.rev !signals
    | tick :: rest ->
      update_state agent.state tick agent.genome agent.neural_net;
      update_runner_mode agent.state agent.genome last_trade_result;
      
      let should_enter_long = 
        agent.state.ema_fast > agent.state.ema_slow &&
        agent.state.rsi < agent.genome.rsi_threshold &&
        agent.state.momentum > 0.0 &&
        agent.state.signal_strength > 0.3 &&
        agent.state.price > agent.state.fib_support
      in
      
      let should_enter_short = 
        agent.state.ema_fast < agent.state.ema_slow &&
        agent.state.rsi > (100.0 -. agent.genome.rsi_threshold) &&
        agent.state.momentum < 0.0 &&
        agent.state.signal_strength < -0.3 &&
        agent.state.price < agent.state.fib_resistance
      in
      
      let should_hold_runner = 
        agent.state.runner_mode &&
        agent.state.runner_score > agent.genome.runner_exit_threshold
      in
      
      if should_enter_long && not should_hold_runner then (
        let take_profit = if agent.state.runner_mode then 
          tick.last *. agent.genome.take_profit_mult *. 1.5
        else 
          tick.last *. agent.genome.take_profit_mult in
        let stop_loss = tick.last *. agent.genome.stop_loss_mult in
        signals := ("LONG", tick.last, take_profit, stop_loss, agent.state.runner_score) :: !signals;
        process_ticks rest 1.0
      ) else if should_enter_short && not should_hold_runner then (
        let take_profit = if agent.state.runner_mode then 
          tick.last /. agent.genome.take_profit_mult /. 1.5
        else 
          tick.last /. agent.genome.take_profit_mult in
        let stop_loss = tick.last /. agent.genome.stop_loss_mult in
        signals := ("SHORT", tick.last, take_profit, stop_loss, agent.state.runner_score) :: !signals;
        process_ticks rest 1.0
      ) else
        process_ticks rest 0.0
  in
  process_ticks ticks 0.0

(* Fitness evaluation with multi-objective optimization *)
let evaluate_agent agent ticks =
  let signals = generate_signals ticks agent in
  let total_trades = List.length signals in
  let profitable_trades = List.fold_left (fun acc (_, entry, tp, sl, _) ->
    let win_prob = min 0.8 (agent.state.signal_strength *. 0.1 +. 0.5) in
    if Random.float 1.0 < win_prob then acc + 1 else acc
  ) 0 signals in
  
  let win_rate = if total_trades > 0 then float_of_int profitable_trades /. float_of_int total_trades else 0.0 in
  let total_profit = List.fold_left (fun acc (_, entry, tp, sl, runner_score) ->
    let win_prob = min 0.8 (runner_score *. 0.1 +. 0.5) in
    let outcome = if Random.float 1.0 < win_prob then tp -. entry else sl -. entry in
    acc +. outcome
  ) 0.0 signals in
  
  agent.total_pnl <- agent.total_pnl +. total_profit;
  agent.trade_count <- agent.trade_count + total_trades;
  agent.win_count <- agent.win_count + profitable_trades;
  
  (* Multi-objective fitness *)
  let avg_return = if total_trades > 0 then total_profit /. float_of_int total_trades else 0.0 in
  let sharpe = if agent.state.volatility > 0.0 then avg_return /. agent.state.volatility else 0.0 in
  let reward_risk = if total_trades > 0 then win_rate *. 2.0 -. 1.0 else 0.0 in
  
  agent.fitness <- (avg_return *. 0.4) +. (sharpe *. 0.3) +. (reward_risk *. 0.3);
  
  (* Update neural network fitness history *)
  agent.neural_net.fitness_history <- agent.fitness :: agent.neural_net.fitness_history;
  if List.length agent.neural_net.fitness_history > 10 then
    agent.neural_net.fitness_history <- take 10 agent.neural_net.fitness_history;
  
  agent

(* Genetic operations *)
let crossover g1 g2 = {
  alpha_fast = if Random.bool () then g1.alpha_fast else g2.alpha_fast;
  alpha_slow = if Random.bool () then g1.alpha_slow else g2.alpha_slow;
  rsi_threshold = (g1.rsi_threshold +. g2.rsi_threshold) /. 2.0;
  rsi_period = if Random.bool () then g1.rsi_period else g2.rsi_period;
  take_profit_mult = (g1.take_profit_mult +. g2.take_profit_mult) /. 2.0;
  stop_loss_mult = (g1.stop_loss_mult +. g2.stop_loss_mult) /. 2.0;
  fib_ratio1 = (g1.fib_ratio1 +. g2.fib_ratio1) /. 2.0;
  fib_ratio2 = (g1.fib_ratio2 +. g2.fib_ratio2) /. 2.0;
  reward_risk_weight = (g1.reward_risk_weight +. g2.reward_risk_weight) /. 2.0;
  ode_accel_coeff = (g1.ode_accel_coeff +. g2.ode_accel_coeff) /. 2.0;
  ode_jerk_coeff = (g1.ode_jerk_coeff +. g2.ode_jerk_coeff) /. 2.0;
  runner_hold_threshold = (g1.runner_hold_threshold +. g2.runner_hold_threshold) /. 2.0;
  runner_exit_threshold = (g1.runner_exit_threshold +. g2.runner_exit_threshold) /. 2.0;
  volatility_sensitivity = (g1.volatility_sensitivity +. g2.volatility_sensitivity) /. 2.0;
  fibonacci_targets = Array.map2 (fun a b -> (a +. b) /. 2.0) g1.fibonacci_targets g2.fibonacci_targets;
  neural_depth = if Random.bool () then g1.neural_depth else g2.neural_depth;
  neural_width = if Random.bool () then g1.neural_width else g2.neural_width;
  preferred_activation = if Random.bool () then g1.preferred_activation else g2.preferred_activation;
  runner_patience = (g1.runner_patience +. g2.runner_patience) /. 2.0;
  risk_adaptation = (g1.risk_adaptation +. g2.risk_adaptation) /. 2.0;
}

let mutate genome = {
  genome with
  alpha_fast = genome.alpha_fast *. (1.0 +. (Random.float 0.2 -. 0.1));
  alpha_slow = genome.alpha_slow *. (1.0 +. (Random.float 0.2 -. 0.1));
  rsi_threshold = max 10.0 (min 90.0 (genome.rsi_threshold +. (Random.float 10.0 -. 5.0)));
  volatility_sensitivity = max 0.1 (min 3.0 (genome.volatility_sensitivity *. (1.0 +. (Random.float 0.2 -. 0.1))));
  runner_hold_threshold = max 0.1 (min 2.0 (genome.runner_hold_threshold *. (1.0 +. (Random.float 0.2 -. 0.1))));
  runner_exit_threshold = max 0.1 (min 1.5 (genome.runner_exit_threshold *. (1.0 +. (Random.float 0.2 -. 0.1))));
  preferred_activation = all_activations.(Random.int (Array.length all_activations));
  runner_patience = max 0.5 (min 5.0 (genome.runner_patience *. (1.0 +. (Random.float 0.2 -. 0.1))));
  risk_adaptation = max 0.1 (min 2.0 (genome.risk_adaptation *. (1.0 +. (Random.float 0.2 -. 0.1))));
}

(* Agent creation *)
let create_agent id generation =
  let default_genome = {
    alpha_fast = 0.15 +. Random.float 0.1;
    alpha_slow = 0.03 +. Random.float 0.04;
    rsi_threshold = 25.0 +. Random.float 10.0;
    rsi_period = 10 + Random.int 10;
    take_profit_mult = 1.005 +. Random.float 0.015;
    stop_loss_mult = 0.995 -. Random.float 0.01;
    fib_ratio1 = 0.382; fib_ratio2 = 0.618;
    reward_risk_weight = 1.0 +. Random.float 1.0;
    ode_accel_coeff = 0.05 +. Random.float 0.1;
    ode_jerk_coeff = 0.005 +. Random.float 0.01;
    runner_hold_threshold = 0.8 +. Random.float 0.4;
    runner_exit_threshold = 0.4 +. Random.float 0.4;
    volatility_sensitivity = 0.5 +. Random.float 1.0;
    fibonacci_targets = [| 0.236; 0.382; 0.5; 0.618; 0.786 |];
    neural_depth = 2 + Random.int 3;
    neural_width = 8 + Random.int 8;
    preferred_activation = all_activations.(Random.int (Array.length all_activations));
    runner_patience = 1.0 +. Random.float 2.0;
    risk_adaptation = 0.5 +. Random.float 1.0;
  } in
  
  let neural_net = create_neural_network default_genome.neural_depth default_genome.neural_width 8 in
  let init_weights = Array.init (Array.length all_activations) (fun _ -> Random.float 0.1) in
  let activation_policy = { 
    weights = init_weights; 
    history = []; 
    performance_map = [] 
  } in
  let comm = {
    id = id;
    inbox = [];
    outbox = [];
    knowledge_base = [];
    trust_scores = List.init 50 (fun j -> (j, 0.5));
    memory_decay_rate = 0.01 +. Random.float 0.02;
    message_filter = default_filter;
  } in
  
  {
    id = id;
    genome = default_genome;
    neural_net = neural_net;
    fitness = 0.0;
    state = init_state 100.0;
    total_pnl = 0.0;
    trade_count = 0;
    win_count = 0;
    activation_policy = activation_policy;
    comms = comm;
    generation = generation;
    runner_trades = [];
    adaptation_rate = 0.1 +. Random.float 0.1;
  }

(* Population evolution with sequential processing *)
let evolve population ticks generation =
  Printf.printf "\n🧬 GENERATION %d EVOLUTION 🧬\n" generation;
  Printf.printf "=====================================\n";
  
  (* Sequential evaluation *)
  let evaluated = List.map (fun a -> evaluate_agent a ticks) population in
  
  (* Sort by fitness *)
  let sorted = List.sort (fun a b -> compare b.fitness a.fitness) evaluated in
  
  (* Print top performers *)
  Printf.printf "\n🏆 TOP PERFORMERS:\n";
  let top_3 = take 3 sorted in
  List.iteri (fun i agent ->
    Printf.printf "Rank %d: Agent %d - Fitness=%.6f, PnL=%.4f, Trades=%d, Win Rate=%.2f%%\n" 
      (i+1) agent.id agent.fitness agent.total_pnl agent.trade_count 
      (if agent.trade_count > 0 then float_of_int agent.win_count /. float_of_int agent.trade_count *. 100.0 else 0.0);
    Printf.printf "  Runner Mode: %s, Runner Score: %.4f\n"
      (if agent.state.runner_mode then "ACTIVE" else "INACTIVE") agent.state.runner_score;
    Printf.printf "  Neural Depth: %d, Width: %d, Activation: %s\n"
      agent.genome.neural_depth agent.genome.neural_width
      (match agent.genome.preferred_activation with
       | Relu -> "Relu" | LeakyRelu -> "LeakyRelu" | Tanh -> "Tanh" 
       | Swish -> "Swish" | Mish -> "Mish" | Sigmoid -> "Sigmoid"
       | GELU -> "GELU" | ELU -> "ELU" | Adaptive -> "Adaptive");
  ) top_3;
  
  (* Elitism: keep top 10% *)
  let elite = take 10 sorted in
  
  (* Generate offspring *)
  let offspring = List.init (List.length population - 10) (fun i ->
    let p1 = List.nth elite (Random.int (List.length elite)) in
    let p2 = List.nth elite (Random.int (List.length elite)) in
    let child_genome = mutate (crossover p1.genome p2.genome) in
    let child = create_agent (List.length population + i) (generation + 1) in
    { child with genome = child_genome }
  ) in
  
  (* Process communication *)
  List.iter (fun agent -> process_inbox agent.comms) elite;
  
  elite @ offspring

(* Initialize population *)
let initialize_population size =
  List.init size (fun i -> create_agent i 0)

(* Generate synthetic tick data *)
let generate_ticks num_ticks base_price =
  let rec gen_ticks acc count price =
    if count <= 0 then List.rev acc
    else
      let noise = (Random.float 0.004) -. 0.002 in
      let trend = sin (float_of_int (num_ticks - count) *. 0.01) *. 0.001 in
      let new_price = price *. (1.0 +. noise +. trend) in
      let spread = new_price *. 0.0001 in
      let tick = {
        bid = new_price -. spread;
        ask = new_price +. spread;
        last = new_price;
        volume = 1000.0 +. Random.float 5000.0;
        timestamp = float_of_int (num_ticks - count);
      } in
      gen_ticks (tick :: acc) (count - 1) new_price
  in
  gen_ticks [] num_ticks base_price

(* Main evolution loop *)
let () =
  Random.self_init ();
  
  Printf.printf "🔥🔥🔥 SIGNALNET EVOLUTIONARY TRADING SYSTEM 🔥🔥🔥\n";
  Printf.printf "============================================================\n";
  Printf.printf "🧬 Multi-Agent Genetic Neural Evolution with Runner Intelligence\n";
  Printf.printf "🎯 Advanced Activation Function Optimization & Communication\n";
  Printf.printf "📊 Fibonacci Analysis & Risk-Reward Optimization\n\n";
  
  let ticks = generate_ticks 5000 100.0 in
  Printf.printf "📈 Generated %d market ticks for simulation\n" (List.length ticks);
  
  let population_size = 50 in
  let initial_population = initialize_population population_size in
  Printf.printf "🏭 Initialized population of %d agents\n\n" population_size;
  
  let generations = 10 in
  let final_population = ref initial_population in
  
  for gen = 0 to generations - 1 do
    final_population := evolve !final_population ticks gen;
    
    let best_agent = List.hd (List.sort (fun a b -> compare b.fitness a.fitness) !final_population) in
    Printf.printf "\n📊 GENERATION %d SUMMARY:\n" gen;
    Printf.printf "Best Fitness: %.6f\n" best_agent.fitness;
    Printf.printf "Best PnL: %.4f\n" best_agent.total_pnl;
    Printf.printf "Runner Score: %.4f\n" best_agent.state.runner_score;
    Printf.printf "Neural Complexity: %dx%d\n" best_agent.genome.neural_depth best_agent.genome.neural_width;
    Printf.printf "===================================\n\n";
  done;
  
  let best_agent = List.hd (List.sort (fun a b -> compare b.fitness a.fitness) !final_population) in
  Printf.printf "\n🎉 EVOLUTION COMPLETE! 🎉\n";
  Printf.printf "🏆 CHAMPION AGENT: #%d\n" best_agent.id;
  Printf.printf "💰 Final PnL: %.4f\n" best_agent.total_pnl;
  Printf.printf "🎯 Final Fitness: %.6f\n" best_agent.fitness;
  Printf.printf "🔥 Total Trades: %d\n" best_agent.trade_count;
  Printf.printf "📈 Win Rate: %.2f%%\n" 
    (if best_agent.trade_count > 0 then float_of_int best_agent.win_count /. float_of_int best_agent.trade_count *. 100.0 else 0.0);
  Printf.printf "🏃 Runner Intelligence: %.4f\n" best_agent.state.runner_score;
  Printf.printf "🧠 Neural Architecture: %dx%d layers\n" best_agent.genome.neural_depth best_agent.genome.neural_width;
  Printf.printf "\n🌟 SIGNALNET EVOLUTION COMPLETE! 🌟\n"
