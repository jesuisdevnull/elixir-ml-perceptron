defmodule Utils do
  def random_float(lower,upper) do
    :rand.uniform() * (lower - upper) + upper
  end

  def sigmoid(x) do
    1/(1+:math.exp(-x))
  end
end

defmodule VecOps do
  def dot(inputs,weights) when length(inputs) == length(weights) do
    Enum.zip(inputs, weights) |> Enum.map(fn {a, b} -> a*b end) |> Enum.sum
  end

  def add(a,b) when length(a) == length(b) do
    Enum.zip(a,b) |> Enum.map(fn {a,b} -> a+b end)
  end

  def scale(a,b) when is_list(a) do
    Enum.map(a, fn x -> x * b end)
  end
end

defmodule Neuron do
  defstruct [:id, :weights, :bias]

  def create(id, num_weights) when is_integer(num_weights) do
    w = for _ <- 1..num_weights, do: Utils.random_float(-5,5)
    b = Utils.random_float(0,1)
    %Neuron{id: id, weights: w, bias: b}
  end

  def activate(neuron, input) when length(input) == length(neuron.weights) do
    VecOps.dot(neuron.weights, input) |> (Kernel.+neuron.bias) |> Utils.sigmoid
  end

  def update_weights(neuron, input, exp_out, act_out, rate) do
    delta = for x <- input, do: rate * (exp_out-act_out) * x
    nws = VecOps.add(neuron.weights, delta)
    %{neuron | weights: nws}
  end

end

defmodule NeuralNet do
  defstruct [:neurons, rate: 0.07, margin: 0.02]

  def create(num_neurons, num_inputs) when is_integer(num_neurons) do
    ns = for n <- 1..num_neurons, do: Neuron.create(n,num_inputs)
    %NeuralNet{neurons: ns}
  end

  def feed_forward(net, input) do
    for neur <- net.neurons, do: Neuron.activate(neur, input)
  end

  def train(net, input, label) do
    output = NeuralNet.feed_forward(net, input)
    labeled_outputs = Enum.zip([net.neurons, label,output]) 
    updated_neurons = for {neuron, expected, actual} <- labeled_outputs, do: Neuron.update_weights(neuron, input, expected, actual, net.rate)
    %{net | neurons: updated_neurons}
  end

  def train(net, input, label, epochs) do
    unless epochs == 0 do
      new_net = train(net, input, label)
      train(new_net, input,label,epochs-1)
    else
      train(net, input, label)
    end
  end

end

input = [[0,1,0,0,0,1,1,0,0,1],[1,0,0,0,0]]
nn = NeuralNet.create(5,10)
f_out = NeuralNet.feed_forward(nn,Enum.at(input, 0))
f_out |> IO.inspect
nn = NeuralNet.train(nn, Enum.at(input,0), Enum.at(input,1), 500)
l_out =  NeuralNet.feed_forward(nn,Enum.at(input, 0))
l_out |> IO.inspect
