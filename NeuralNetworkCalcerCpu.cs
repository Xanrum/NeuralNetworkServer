using System.Numerics;
using System.Runtime.Intrinsics;

namespace NeuralNetworkServer;

public class NeuralNetworkCalcerCpu
{


    public float[][] Calc(float[] inputs, int[] indexes, float[] synapses, int[] model)
    {
        var result = new float[indexes.Length][];
        var max = model.Max();
        Span<float> prev = stackalloc float[max];
        Span<float> next = stackalloc float[max];
        for (int index = 0; index < indexes.Length; index++)
        {
            var startInput = indexes[index];
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
            result[index] = r;
        }
        return result;
    }
}