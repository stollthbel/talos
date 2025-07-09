(* SignalNet: Self-Expanding Multi-Agent Evolutionary Neural ODE System *)

(* Tick data structure *)
type tick = {
  bid: float;
  ask: float;
  last: float;
  volume: float;
  timestamp: float;
}

(* ODE-based state tracking *)
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
}

(* Activation functions *)
type activation_fn = Relu | LeakyRelu | Tanh | Swish | Mish | Sigmoid

(* Core genome and agent definitions — the base for evolutionary logic *)
type genome = {
  alpha_fast: float;
  alpha_slow: float;
  rsi_threshold: float;
  take_profit_mult: float;
  stop_loss_mult: float;
  fib_ratio1: float;
  fib_ratio2: float;
  reward_risk_weight: float;
  ode_accel_coeff: float;
  ode_jerk_coeff: float;
  runner_hold_threshold: float;
  runner_exit_threshold: float;
  preferred_activation: activation_fn;
}

(* Agent's mutable neural representation *)
type neuron = {
  id: int;
  weights: float array;
  bias: float;
  mutable contribution: float;
}

type layer = neuron array

type brain = {
  mutable layers: layer list;
}

(* Mutation and growth rules for brain *)
let grow_neuron id input_size = {
  id;
  weights = Array.init input_size (fun _ -> Random.float 0.1 -. 0.05);
  bias = Random.float 0.1 -. 0.05;
  contribution = 0.0;
}

let grow_layer size input_size =
  Array.init size (fun i -> grow_neuron i input_size)

let add_layer brain input_size =
  let size = 4 + Random.int 4 in
  brain.layers <- brain.layers @ [grow_layer size input_size]

let prune_dead_neurons brain threshold =
  brain.layers <- List.map (fun layer ->
    Array.filter (fun n -> n.contribution > threshold) layer
  ) brain.layers

(* Signal-focused agent *)
type agent = {
  id: int;
  genome: genome;
  brain: brain;
  mutable fitness: float;
  mutable info_gain: float;
  mutable entropy_reduction: float;
  mutable trade_history: float list;
  mutable state: state;
  mutable total_pnl: float;
  mutable trade_count: int;
  mutable activation_performance: (activation_fn * float) list;
  mutable knowledge_base: (string * float) list;
  mutable generation: int;
}

(* Create initial state *)
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
}

(* Generate synthetic tick data *)
let generate_ticks num_ticks base_price =
  let rec gen_ticks acc count price =
    if count <= 0 then List.rev acc
    else
      let noise = (Random.float 0.02) -. 0.01 in
      let new_price = price *. (1.0 +. noise) in
      let spread = new_price *. 0.0001 in
      let tick = {
        bid = new_price -. spread;
        ask = new_price +. spread;
        last = new_price;
        volume = 100.0 +. Random.float 1000.0;
        timestamp = float_of_int (1000 - count);
      } in
      gen_ticks (tick :: acc) (count - 1) new_price
  in
  gen_ticks [] num_ticks base_price

(* Update state with ODE-based evolution *)
let update_state state tick genome =
  let dt = 1.0 in
  let price_change = tick.last -. state.price in
  
  (* Update EMAs *)
  state.ema_fast <- state.ema_fast +. genome.alpha_fast *. (tick.last -. state.ema_fast);
  state.ema_slow <- state.ema_slow +. genome.alpha_slow *. (tick.last -. state.ema_slow);
  
  (* Update momentum and higher-order derivatives *)
  let old_momentum = state.momentum in
  let old_acceleration = state.acceleration in
  
  state.momentum <- price_change /. dt;
  state.acceleration <- (state.momentum -. old_momentum) /. dt;
  state.jerk <- (state.acceleration -. old_acceleration) /. dt;
  
  (* Apply ODE coefficients *)
  state.acceleration <- state.acceleration *. genome.ode_accel_coeff;
  state.jerk <- state.jerk *. genome.ode_jerk_coeff;
  
  (* Update RSI components *)
  if price_change > 0.0 then begin
    state.gain <- state.gain *. 0.93 +. price_change *. 0.07;
    state.loss <- state.loss *. 0.93;
  end else begin
    state.gain <- state.gain *. 0.93;
    state.loss <- state.loss *. 0.93 +. (-.price_change) *. 0.07;
  end;
  
  (* Calculate RSI *)
  let rs = if state.loss > 0.0 then state.gain /. state.loss else 100.0 in
  state.rsi <- 100.0 -. (100.0 /. (1.0 +. rs));
  
  (* Update volume EMA *)
  state.volume_ema <- state.volume_ema *. 0.9 +. tick.volume *. 0.1;
  
  (* Update price *)
  state.price <- tick.last;
  
  (* Calculate signal strength *)
  let ema_divergence = abs_float (state.ema_fast -. state.ema_slow) /. state.price in
  let momentum_strength = abs_float state.momentum in
  state.signal_strength <- ema_divergence *. momentum_strength *. (abs_float state.acceleration)

(* Utility functions *)
let rec take n lst =
  if n <= 0 then []
  else match lst with
  | [] -> []
  | h :: t -> h :: take (n - 1) t

(* Neural network forward pass *)
let forward_pass brain inputs =
  let rec process_layers acc layers =
    match layers with
    | [] -> acc
    | layer :: rest ->
      let layer_output = Array.mapi (fun i neuron ->
        let weighted_sum = Array.fold_left2 (+.) neuron.bias 
          (Array.map2 ( *. ) neuron.weights acc) acc in
        let activated = tanh weighted_sum in
        neuron.contribution <- abs_float activated;
        activated
      ) layer in
      process_layers layer_output rest
  in
  if List.length brain.layers = 0 then inputs
  else process_layers inputs brain.layers

(* Enhanced signal generation with neural network *)
let generate_neural_signals ticks genome brain =
  if List.length ticks = 0 then []
  else
    let first_tick = List.hd ticks in
    let state = init_state first_tick.last in
    
    let rec process_ticks acc remaining_ticks =
      match remaining_ticks with
      | [] -> List.rev acc
      | tick :: rest ->
        update_state state tick genome;
        
        (* Create neural network input *)
        let nn_input = [| state.ema_fast; state.ema_slow; state.rsi; state.momentum; 
                         state.acceleration; state.jerk; state.signal_strength |] in
        
        (* Get neural network output *)
        let nn_output = forward_pass brain nn_input in
        let neural_signal = if Array.length nn_output > 0 then nn_output.(0) else 0.0 in
        
        (* Generate signal based on neural network and traditional conditions *)
        let signal = 
          if neural_signal > 0.3 && state.ema_fast > state.ema_slow && state.rsi < genome.rsi_threshold then
            let take_profit = tick.last *. genome.take_profit_mult in
            let stop_loss = tick.last /. genome.stop_loss_mult in
            let signal_quality = state.signal_strength *. genome.reward_risk_weight *. neural_signal in
            Some (tick.bid, tick.ask, tick.last, take_profit, stop_loss, signal_quality)
          else if neural_signal < -0.3 && state.ema_fast < state.ema_slow && state.rsi > (100.0 -. genome.rsi_threshold) then
            let take_profit = tick.last /. genome.take_profit_mult in
            let stop_loss = tick.last *. genome.stop_loss_mult in
            let signal_quality = state.signal_strength *. genome.reward_risk_weight *. abs_float neural_signal in
            Some (tick.bid, tick.ask, tick.last, take_profit, stop_loss, signal_quality)
          else
            None
        in
        
        match signal with
        | Some s -> process_ticks (s :: acc) rest
        | None -> process_ticks acc rest
    in
    
    process_ticks [] ticks

(* Generate trading signals *)
let generate_signals ticks genome =
  if List.length ticks = 0 then []
  else
    let first_tick = List.hd ticks in
    let state = init_state first_tick.last in
    
    let rec process_ticks acc remaining_ticks =
      match remaining_ticks with
      | [] -> List.rev acc
      | tick :: rest ->
        update_state state tick genome;
        
        (* Generate signal based on conditions *)
        let signal = 
          if state.ema_fast > state.ema_slow && state.rsi < genome.rsi_threshold then
            (* Buy signal *)
            let take_profit = tick.last *. genome.take_profit_mult in
            let stop_loss = tick.last /. genome.stop_loss_mult in
            let signal_quality = state.signal_strength *. genome.reward_risk_weight in
            Some (tick.bid, tick.ask, tick.last, take_profit, stop_loss, signal_quality)
          else if state.ema_fast < state.ema_slow && state.rsi > (100.0 -. genome.rsi_threshold) then
            (* Sell signal *)
            let take_profit = tick.last /. genome.take_profit_mult in
            let stop_loss = tick.last *. genome.stop_loss_mult in
            let signal_quality = state.signal_strength *. genome.reward_risk_weight in
            Some (tick.bid, tick.ask, tick.last, take_profit, stop_loss, signal_quality)
          else
            None
        in
        
        match signal with
        | Some s -> process_ticks (s :: acc) rest
        | None -> process_ticks acc rest
    in
    
    process_ticks [] ticks

(* Evaluate agent performance *)
let evaluate_agent_performance agent ticks =
  let signals = generate_signals ticks agent.genome in
  let total_profit = List.fold_left (fun acc (_, _, entry, tp, sl, quality) ->
    (* Simulate trade outcome based on quality *)
    let win_prob = min 0.8 (quality *. 0.1 +. 0.4) in
    let outcome = if Random.float 1.0 < win_prob then tp -. entry else sl -. entry in
    acc +. outcome
  ) 0.0 signals in
  
  agent.total_pnl <- agent.total_pnl +. total_profit;
  agent.trade_count <- agent.trade_count + List.length signals;
  
  (* Update trade history *)
  let new_returns = List.map (fun (_, _, entry, tp, sl, _) ->
    let win_prob = 0.6 in
    if Random.float 1.0 < win_prob then (tp -. entry) /. entry else (sl -. entry) /. entry
  ) signals in
  agent.trade_history <- new_returns @ agent.trade_history;
  
  (* Calculate fitness based on total PnL and consistency *)
  let avg_return = if agent.trade_count > 0 then agent.total_pnl /. float_of_int agent.trade_count else 0.0 in
  let consistency = 1.0 -. agent.entropy_reduction in
  agent.fitness <- avg_return *. consistency *. agent.genome.reward_risk_weight;
  
  total_profit

(* Enhanced crossover with runner parameters *)
let crossover g1 g2 = {
  alpha_fast = (g1.alpha_fast +. g2.alpha_fast) /. 2.;
  alpha_slow = (g1.alpha_slow +. g2.alpha_slow) /. 2.;
  rsi_threshold = (g1.rsi_threshold +. g2.rsi_threshold) /. 2.;
  take_profit_mult = (g1.take_profit_mult +. g2.take_profit_mult) /. 2.;
  stop_loss_mult = (g1.stop_loss_mult +. g2.stop_loss_mult) /. 2.;
  fib_ratio1 = (g1.fib_ratio1 +. g2.fib_ratio1) /. 2.;
  fib_ratio2 = (g1.fib_ratio2 +. g2.fib_ratio2) /. 2.;
  reward_risk_weight = (g1.reward_risk_weight +. g2.reward_risk_weight) /. 2.;
  ode_accel_coeff = (g1.ode_accel_coeff +. g2.ode_accel_coeff) /. 2.;
  ode_jerk_coeff = (g1.ode_jerk_coeff +. g2.ode_jerk_coeff) /. 2.;
  runner_hold_threshold = (g1.runner_hold_threshold +. g2.runner_hold_threshold) /. 2.;
  runner_exit_threshold = (g1.runner_exit_threshold +. g2.runner_exit_threshold) /. 2.;
  preferred_activation = if Random.bool () then g1.preferred_activation else g2.preferred_activation;
}

(* Enhanced mutation with runner parameters *)
let mutate_genome g = {
  g with
  alpha_fast = g.alpha_fast *. (0.9 +. Random.float 0.2);
  alpha_slow = g.alpha_slow *. (0.9 +. Random.float 0.2);
  reward_risk_weight = g.reward_risk_weight *. (0.95 +. Random.float 0.1);
  runner_hold_threshold = g.runner_hold_threshold *. (0.9 +. Random.float 0.2);
  runner_exit_threshold = g.runner_exit_threshold *. (0.9 +. Random.float 0.2);
  preferred_activation = match Random.int 6 with
    | 0 -> Relu | 1 -> LeakyRelu | 2 -> Tanh | 3 -> Swish | 4 -> Mish | _ -> Sigmoid
}

(* Entropy and signal quality estimator placeholder *)
let estimate_entropy trades =
  let n = List.length trades in
  if n < 2 then 1.0 else
    let mean = List.fold_left (+.) 0.0 trades /. float n in
    let var = List.fold_left (fun acc x -> acc +. (x -. mean) ** 2.) 0.0 trades /. float n in
    1.0 /. (1.0 +. var)

let estimate_info_gain agent =
  let entropy = estimate_entropy agent.trade_history in
  let gain = entropy *. agent.fitness in
  agent.entropy_reduction <- entropy;
  agent.info_gain <- gain;
  gain

(* Decision to evolve *)
let evolve_agent agent =
  let gain = estimate_info_gain agent in
  if gain > 0.1 then add_layer agent.brain 5;
  prune_dead_neurons agent.brain 0.01

(* Activation function implementations *)
let activate fn x =
  match fn with
  | Relu -> max 0.0 x
  | LeakyRelu -> if x > 0.0 then x else 0.01 *. x
  | Tanh -> tanh x
  | Swish -> x *. (1.0 /. (1.0 +. exp (-.x)))
  | Mish -> x *. tanh (log (1.0 +. exp x))
  | Sigmoid -> 1.0 /. (1.0 +. exp (-.x))

(* String representation of activation functions *)
let string_of_activation fn =
  match fn with
  | Relu -> "Relu"
  | LeakyRelu -> "LeakyRelu"
  | Tanh -> "Tanh"
  | Swish -> "Swish"
  | Mish -> "Mish"
  | Sigmoid -> "Sigmoid"

(* Runner mode logic *)
let update_runner_mode state genome last_trade_result =
  (* Update consecutive wins/losses *)
  if last_trade_result > 0.0 then
    state.consecutive_wins <- state.consecutive_wins + 1
  else
    state.consecutive_wins <- 0;
  
  (* Calculate runner score based on momentum and wins *)
  let momentum_factor = abs_float state.momentum in
  let win_factor = float_of_int state.consecutive_wins in
  state.runner_score <- momentum_factor *. win_factor *. state.signal_strength;
  
  (* Enter runner mode if score exceeds threshold *)
  if state.runner_score > genome.runner_hold_threshold then
    state.runner_mode <- true
  else if state.runner_score < genome.runner_exit_threshold then
    state.runner_mode <- false

(* Enhanced signal generation with runner intelligence *)
let generate_runner_signals ticks genome =
  if List.length ticks = 0 then []
  else
    let first_tick = List.hd ticks in
    let state = init_state first_tick.last in
    
    let rec process_ticks acc remaining_ticks current_position =
      match remaining_ticks with
      | [] -> List.rev acc
      | tick :: rest ->
        update_state state tick genome;
        
        (* Simulate trade result for runner mode *)
        let last_trade_result = if Random.float 1.0 > 0.5 then 1.0 else -1.0 in
        update_runner_mode state genome last_trade_result;
        
        (* Generate signal based on runner mode *)
        let signal = 
          if state.runner_mode && current_position <> None then
            (* In runner mode - hold position longer *)
            state.position_hold_time <- state.position_hold_time +. 1.0;
            None
          else if state.ema_fast > state.ema_slow && state.rsi < genome.rsi_threshold then
            (* Buy signal *)
            let take_profit = if state.runner_mode then 
              tick.last *. genome.take_profit_mult *. 1.5 (* Extended TP in runner mode *)
            else 
              tick.last *. genome.take_profit_mult in
            let stop_loss = tick.last /. genome.stop_loss_mult in
            let signal_quality = state.signal_strength *. genome.reward_risk_weight in
            Some (tick.bid, tick.ask, tick.last, take_profit, stop_loss, signal_quality)
          else if state.ema_fast < state.ema_slow && state.rsi > (100.0 -. genome.rsi_threshold) then
            (* Sell signal *)
            let take_profit = if state.runner_mode then 
              tick.last /. genome.take_profit_mult /. 1.5 (* Extended TP in runner mode *)
            else 
              tick.last /. genome.take_profit_mult in
            let stop_loss = tick.last *. genome.stop_loss_mult in
            let signal_quality = state.signal_strength *. genome.reward_risk_weight in
            Some (tick.bid, tick.ask, tick.last, take_profit, stop_loss, signal_quality)
          else
            None
        in
        
        match signal with
        | Some s -> process_ticks (s :: acc) rest (Some s)
        | None -> process_ticks acc rest current_position
    in
    
    process_ticks [] ticks None

(* Performance tracking for activation functions *)
let track_activation_performance agent activation_fn performance =
  let existing = List.assoc_opt activation_fn agent.activation_performance in
  match existing with
  | Some old_perf -> 
    let new_perf = (old_perf +. performance) /. 2.0 in
    agent.activation_performance <- (activation_fn, new_perf) :: 
      (List.remove_assoc activation_fn agent.activation_performance)
  | None ->
    agent.activation_performance <- (activation_fn, performance) :: agent.activation_performance

(* Knowledge base management *)
let update_knowledge_base agent key value =
  let existing = List.assoc_opt key agent.knowledge_base in
  match existing with
  | Some old_val -> 
    let new_val = (old_val +. value) /. 2.0 in
    agent.knowledge_base <- (key, new_val) :: (List.remove_assoc key agent.knowledge_base)
  | None ->
    agent.knowledge_base <- (key, value) :: agent.knowledge_base

(* Create population and evolve *)
let create_agent id generation =
  let default_genome = {
    alpha_fast = 0.15 +. Random.float 0.1;
    alpha_slow = 0.03 +. Random.float 0.04;
    rsi_threshold = 25.0 +. Random.float 10.0;
    take_profit_mult = 1.01 +. Random.float 0.02;
    stop_loss_mult = 0.98 +. Random.float 0.02;
    fib_ratio1 = 0.382; fib_ratio2 = 0.618;
    reward_risk_weight = 1.0 +. Random.float 1.0;
    ode_accel_coeff = 0.05 +. Random.float 0.1;
    ode_jerk_coeff = 0.005 +. Random.float 0.01;
    runner_hold_threshold = 0.8 +. Random.float 0.4;
    runner_exit_threshold = 0.4 +. Random.float 0.4;
    preferred_activation = match Random.int 6 with
      | 0 -> Relu | 1 -> LeakyRelu | 2 -> Tanh | 3 -> Swish | 4 -> Mish | _ -> Sigmoid;
  } in
  {
    id = id;
    genome = default_genome;
    brain = { layers = [] };
    fitness = 0.0;
    info_gain = 0.0;
    entropy_reduction = 0.0;
    trade_history = [];
    state = init_state 100.0;
    total_pnl = 0.0;
    trade_count = 0;
    activation_performance = [];
    knowledge_base = [];
    generation = generation;
  }

(* Population evolution *)
let evolve_population population ticks generation =
  Printf.printf "\n🧬 GENERATION %d EVOLUTION 🧬\n" generation;
  Printf.printf "=====================================\n";
  
  (* Evaluate all agents *)
  let evaluated_agents = List.mapi (fun i agent ->
    let profit = evaluate_agent_performance agent ticks in
    update_knowledge_base agent "total_profit" profit;
    update_knowledge_base agent "signal_strength" agent.state.signal_strength;
    update_knowledge_base agent "runner_score" agent.state.runner_score;
    track_activation_performance agent agent.genome.preferred_activation agent.fitness;
    
    Printf.printf "Agent %d: PnL=%.4f, Fitness=%.6f, Activation=%s, Runner=%.2f\n" 
      i profit agent.fitness 
      (string_of_activation agent.genome.preferred_activation)
      agent.state.runner_score;
    
    agent
  ) population in
  
  (* Sort by fitness *)
  let sorted_agents = List.sort (fun a b -> compare b.fitness a.fitness) evaluated_agents in
  
  (* Print top performers *)
  Printf.printf "\n🏆 TOP PERFORMERS:\n";
  let top_3 = List.take 3 sorted_agents in
  List.iteri (fun i agent ->
    Printf.printf "Rank %d: Agent %d - Fitness=%.6f, PnL=%.4f, Trades=%d\n" 
      (i+1) agent.id agent.fitness agent.total_pnl agent.trade_count;
    Printf.printf "  Genome: α_fast=%.3f, α_slow=%.3f, RSI=%.1f, Activation=%s\n"
      agent.genome.alpha_fast agent.genome.alpha_slow agent.genome.rsi_threshold
      (string_of_activation agent.genome.preferred_activation);
    Printf.printf "  Runner: Hold=%.2f, Exit=%.2f, Score=%.2f\n"
      agent.genome.runner_hold_threshold agent.genome.runner_exit_threshold 
      agent.state.runner_score;
    Printf.printf "  Knowledge: ";
    List.iter (fun (k, v) -> Printf.printf "%s=%.2f " k v) agent.knowledge_base;
    Printf.printf "\n\n";
  ) top_3;
  
  (* Create next generation *)
  let elite = List.take 4 sorted_agents in
  let offspring = List.init (List.length population - 4) (fun i ->
    let parent1 = List.nth elite (Random.int (List.length elite)) in
    let parent2 = List.nth elite (Random.int (List.length elite)) in
    let child_genome = mutate_genome (crossover parent1.genome parent2.genome) in
    let child = create_agent (List.length population + i) (generation + 1) in
    { child with genome = child_genome }
  ) in
  
  elite @ offspring

(* Print activation function performance summary *)
let print_activation_summary agents =
  Printf.printf "\n🎯 ACTIVATION FUNCTION PERFORMANCE SUMMARY:\n";
  Printf.printf "===========================================\n";
  
  let activation_stats = Hashtbl.create 6 in
  List.iter (fun agent ->
    List.iter (fun (activation, perf) ->
      let current = try Hashtbl.find activation_stats activation with Not_found -> (0, 0.0) in
      let (count, total) = current in
      Hashtbl.replace activation_stats activation (count + 1, total +. perf)
    ) agent.activation_performance
  ) agents;
  
  Hashtbl.iter (fun activation (count, total) ->
    let avg = if count > 0 then total /. float_of_int count else 0.0 in
    Printf.printf "%s: Count=%d, Avg Performance=%.6f\n" 
      (string_of_activation activation) count avg
  ) activation_stats;
  
  Printf.printf "\n"

(* Entry point *)
let () =
  Random.self_init ();
  
  Printf.printf "🔥🔥🔥 SIGNALNET EVOLUTIONARY TRADING SYSTEM 🔥🔥🔥\n";
  Printf.printf "============================================================\n";
  Printf.printf "🧬 Multi-Agent Genetic Neural ODE Evolution Started\n";
  Printf.printf "🎯 Runner Intelligence & Activation Function Optimization\n";
  Printf.printf "📊 Comprehensive Performance Tracking & Knowledge Base\n\n";
  
  (* Generate market data *)
  let ticks = generate_ticks 2000 100.0 in
  Printf.printf "📈 Generated %d market ticks for simulation\n" (List.length ticks);
  
  (* Initialize population *)
  let population_size = 10 in
  let initial_population = List.init population_size (fun i -> create_agent i 0) in
  Printf.printf "🏭 Initialized population of %d agents\n\n" population_size;
  
  (* Run evolution for multiple generations *)
  let generations = 5 in
  let final_population = ref initial_population in
  
  for gen = 0 to generations - 1 do
    final_population := evolve_population !final_population ticks gen;
    
    (* Print generation summary *)
    let best_agent = List.hd (List.sort (fun a b -> compare b.fitness a.fitness) !final_population) in
    Printf.printf "\n📊 GENERATION %d SUMMARY:\n" gen;
    Printf.printf "Best Fitness: %.6f\n" best_agent.fitness;
    Printf.printf "Best PnL: %.4f\n" best_agent.total_pnl;
    Printf.printf "Best Activation: %s\n" (string_of_activation best_agent.genome.preferred_activation);
    Printf.printf "Runner Mode Active: %s\n" (if best_agent.state.runner_mode then "YES" else "NO");
    Printf.printf "===================================\n\n";
    
    (* Evolve agent brains *)
    List.iter (fun agent ->
      evolve_agent agent;
      if List.length agent.brain.layers > 0 then
        Printf.printf "Agent %d: Evolved brain with %d layers\n" agent.id (List.length agent.brain.layers)
    ) !final_population;
  done;
  
  (* Final summary *)
  Printf.printf "\n🎉 EVOLUTION COMPLETE! 🎉\n";
  Printf.printf "=======================\n";
  
  let best_agent = List.hd (List.sort (fun a b -> compare b.fitness a.fitness) !final_population) in
  Printf.printf "🏆 CHAMPION AGENT: #%d\n" best_agent.id;
  Printf.printf "💰 Final PnL: %.4f\n" best_agent.total_pnl;
  Printf.printf "🎯 Final Fitness: %.6f\n" best_agent.fitness;
  Printf.printf "🔥 Total Trades: %d\n" best_agent.trade_count;
  Printf.printf "🧠 Neural Layers: %d\n" (List.length best_agent.brain.layers);
  Printf.printf "⚡ Preferred Activation: %s\n" (string_of_activation best_agent.genome.preferred_activation);
  Printf.printf "🏃 Runner Score: %.4f\n" best_agent.state.runner_score;
  Printf.printf "📚 Knowledge Base Entries: %d\n" (List.length best_agent.knowledge_base);
  
  print_activation_summary !final_population;
  
  (* Test final signals *)
  let test_signals = generate_runner_signals (List.take 20 ticks) best_agent.genome in
  Printf.printf "🚀 Generated %d signals with champion genome\n" (List.length test_signals);
  
  List.iteri (fun i (bid, ask, last, tp, sl, quality) ->
    Printf.printf "Signal %d: Entry=%.4f, TP=%.4f, SL=%.4f, Quality=%.4f\n" 
      i last tp sl quality
  ) test_signals;
  
  Printf.printf "\n🌟 SIGNALNET EVOLUTION COMPLETE! 🌟\n"
