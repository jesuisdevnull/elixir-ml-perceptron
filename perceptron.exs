defmodule Utils do
  def random_float(lower,upper) do
    :rand.uniform() * (lower - upper) + upper
  end

  def sigmoid(x) do
    1/(1+:math.exp(-x))
  end

  def sigmoid_derivative(x) do
    Utils.sigmoid(x)*(1-Utils.sigmoid(x))
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

defmodule SLNeuralNet do
  defstruct [:neurons, rate: 0.07, margin: 0.02]

  def create(num_neurons, num_inputs) when is_integer(num_neurons) do
    ns = for n <- 1..num_neurons, do: Neuron.create(n,num_inputs)
    %SLNeuralNet{neurons: ns}
  end

  def feed_forward(net, input) do
    for neur <- net.neurons, do: Neuron.activate(neur, input)
  end

  def train(net, input, label) do
    output = SLNeuralNet.feed_forward(net, input)
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

defmodule MLNeuralNet do
  defstruct [:layers, rate: 0.07, margin: 0.02]

  def create(neuron_count, layer_count,input_count, output_count) do
    hidden_layers = for layer_n <- 1..layer_count, do:
      for neuron_n <- 1..neuron_count, do:
       (if layer_n == 1 do
          Neuron.create(neuron_n, input_count)
        else
          Neuron.create(neuron_n, neuron_count)
       end)
    output_layer = for n <- 1..output_count, do: Neuron.create(n, neuron_count)
    layers = hidden_layers ++ [output_layer]
    layers |> IO.inspect
    %MLNeuralNet{layers: layers}
  end

  def forward_pass(net, input, _) do
    output_matrix = calculate_layer_output(net.layers, input)
    net_output = List.last(output_matrix)
    if length(net_output) == 1 do
      %{output: hd(net_output), history: output_matrix}
    else
      %{output: net_output, history: output_matrix}
    end
  end

  defp calculate_layer_output(layer_list, _) when layer_list == [] do
    []
  end

  defp calculate_layer_output(layer_list, input) when is_list(layer_list) do
    [layer | rest] = layer_list
    output =  for neur <- layer, do: Neuron.activate(neur, input)
    [output | calculate_layer_output(rest, output)]
  end

end

nn = MLNeuralNet.create(3, 3, 5, 1)
MLNeuralNet.forward_pass(nn,[1,2,3,4,5],0.5) |> IO.inspect
