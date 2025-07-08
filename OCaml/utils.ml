(* utils.ml *)

module Vector = struct
  type t = float array

  let dim v = Array.length v

  let map f v = Array.map f v

  let map2 f v1 v2 =
    Array.init (Array.length v1) (fun i -> f v1.(i) v2.(i))

  let add v1 v2 = map2 ( +. ) v1 v2

  let sub v1 v2 = map2 ( -. ) v1 v2

  let scale a v = map (( *. ) a) v

  let dot v1 v2 =
    Array.fold_left ( +. ) 0.0 (map2 ( *. ) v1 v2)

  let norm v = sqrt (dot v v)

  let print v =
    Printf.printf "[|";
    Array.iter (fun x -> Printf.printf " %.4f;" x) v;
    Printf.printf " |]\n"
end

module Matrix = struct
  type t = float array array

  let dim m = Array.length m, Array.length m.(0)

  let identity n =
    Array.init n (fun i -> Array.init n (fun j -> if i = j then 1.0 else 0.0))

  let transpose m =
    let rows = Array.length m in
    let cols = Array.length m.(0) in
    Array.init cols (fun j -> Array.init rows (fun i -> m.(i).(j)))

  let dot a b =
    let n, m = dim a in
    let _, p = dim b in
    Array.init n (fun i ->
      Array.init p (fun j ->
        let sum = ref 0.0 in
        for k = 0 to m - 1 do
          sum := !sum +. a.(i).(k) *. b.(k).(j)
        done;
        !sum))

  let print m =
    Array.iter (fun row ->
      Array.iter (fun x -> Printf.printf " %.4f " x) row;
      Printf.printf "\n") m
end

(* qr.ml *)

module QR = struct
  open Vector
  open Matrix

  let gram_schmidt a =
    let n, _ = dim a in
    let q = Array.make_matrix n n 0.0 in
    let r = Array.make_matrix n n 0.0 in
    for j = 0 to n - 1 do
      let vj = Array.init n (fun i -> a.(i).(j)) in
      let vj_mod = ref vj in
      for i = 0 to j - 1 do
        let qi = Array.init n (fun k -> q.(k).(i)) in
        let rij = dot qi !vj_mod in
        r.(i).(j) <- rij;
        vj_mod := sub !vj_mod (scale rij qi)
      done;
      let norm = norm !vj_mod in
      r.(j).(j) <- norm;
      for i = 0 to n - 1 do
        q.(i).(j) <- !vj_mod.(i) /. norm
      done
    done;
    q, r
end

(* rk4.ml *)

module RK4 = struct
  open Vector

  let integrate f u0 t0 t_end dt =
    let steps = int_of_float ((t_end -. t0) /. dt) in
    let dim = dim u0 in
    let u = Array.copy u0 in
    let t = ref t0 in
    for _ = 1 to steps do
      let k1 = f !t u in
      let k2 = f (!t +. dt /. 2.0) (add u (scale (dt /. 2.0) k1)) in
      let k3 = f (!t +. dt /. 2.0) (add u (scale (dt /. 2.0) k2)) in
      let k4 = f (!t +. dt) (add u (scale dt k3)) in
      for i = 0 to dim - 1 do
        u.(i) <- u.(i) +. (dt /. 6.0) *. (k1.(i) +. 2.0 *. k2.(i) +. 2.0 *. k3.(i) +. k4.(i))
      done;
      t := !t +. dt
    done;
    u
end

(* ga.ml *)

module GA = struct
  open Vector

  let rand_vec dim =
    Array.init dim (fun _ -> Random.float 2.0 -. 1.0)

  let mutate v rate =
    map (fun x -> x +. (Random.float 2.0 -. 1.0) *. rate) v

  let crossover v1 v2 =
    let alpha = Random.float 1.0 in
    map2 (fun a b -> alpha *. a +. (1.0 -. alpha) *. b) v1 v2

  let evolve ~f ~pop_size ~dim ~generations =
    let pop = Array.init pop_size (fun _ -> rand_vec dim) in
    let fitness v = -. f v in
    for _ = 1 to generations do
      Array.sort (fun a b -> compare (fitness a) (fitness b)) pop;
      for i = pop_size / 2 to pop_size - 1 do
        let p1 = pop.(Random.int (pop_size / 2)) in
        let p2 = pop.(Random.int (pop_size / 2)) in
        let child = crossover p1 p2 |> mutate ~rate:0.1 in
        pop.(i) <- child
      done
    done;
    Array.sort (fun a b -> compare (fitness a) (fitness b)) pop;
    pop.(0)
end