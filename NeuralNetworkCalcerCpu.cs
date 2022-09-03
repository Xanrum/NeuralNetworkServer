using System.Numerics;
using System.Runtime.Intrinsics;

namespace NeuralNetworkServer;

public class NeuralNetworkCalcerCpu
{
    public async Task<float[][]> Calc(float[] inputs, int[] indexes, float[] synapses, int[] model)
    {
        var outputs = new float[indexes.Length][];
        var outputsPerTask = 1000;
        var batchCount = indexes.Length / outputsPerTask;
        var tasks = new List<Task>();
        for (var i = 0; i < batchCount; i++)
        {
            var batchStart = outputsPerTask * i;
            tasks.Add(Task.Run(() => {
                for (int j = 0; j < outputsPerTask; j++)
                {
                    var index = batchStart + j;
                    outputs[index] = SingleCalc(inputs, indexes[index], synapses, model);
                }
            }));
        }
        for (int i = outputsPerTask * batchCount; i < indexes.Length; i++)
        {
            outputs[i] = SingleCalc(inputs, indexes[i], synapses, model);
        }
        await Task.WhenAll(tasks);
        return outputs;
    }

    private float[] SingleCalc(float[] inputs, int startInput, float[] synapses, int[] model)
    {
        var max = model.Max();
        Span<float> prev = stackalloc float[max];
        Span<float> next = stackalloc float[max];
        new Span<float>(inputs, startInput, model[0]).CopyTo(prev);
        var sinapsIndex = 0;
        var vectorLength = Vector<float>.Count;
        for (var i = 1; i < model.Length; i++)
        {
            var prevLen = model[i - 1];
            var len = model[i];
            for (var i1 = 0; i1 < len; i1++)
            {
                var sum = 0f;
                var fromVector = prevLen / vectorLength;
                for (var j = 0; j < fromVector; j++)
                {
                    var prevVector = new Vector<float>(prev.Slice(vectorLength * j, vectorLength));
                    var currentNeuronValues = new Vector<float>(new Span<float>(synapses, sinapsIndex, vectorLength));
                    var dot = Vector.Dot(prevVector, currentNeuronValues);
                    sum += dot;
                    sinapsIndex += vectorLength;
                }
                for (var j = fromVector * vectorLength; j < prevLen; j++)
                {
                    var p = prev[j];
                    var k = p * synapses[sinapsIndex];
                    sum += k;
                    sinapsIndex++;
                }
                next[i1] = MathF.Tanh(sum);
            }
            var tmp = prev;
            prev = next;
            next = tmp;
        }
        var r = new float[model.Last()];
        prev.Slice(0, model.Last()).CopyTo(r);
        return r;
    }
}