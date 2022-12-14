using Microsoft.AspNetCore.Mvc;
using System.Collections.Concurrent;

namespace NeuralNetworkServer.Controllers;

[ApiController]
[Route("[controller]")]
public class NeuralController : ControllerBase
{

    private static readonly ConcurrentDictionary<long, (float[] data, int[] indexes)> InputsCache = new();

    [HttpPost("Load")]
    public void Load(LoadInputsRequest request)
    {
            InputsCache[request.Key] = (request.Data, request.Indexes);
    }
    
    static NeuralNetworkCalcerCuda _calcer = new NeuralNetworkCalcerCuda();
    // static NeuralNetworkCalcerCpu _calcer = new NeuralNetworkCalcerCpu();
        
    [HttpPost("Calc")]
    public async Task<CalcResponse> Calc()
    {
        Request.EnableBuffering();
        Request.Body.Position = 0;
        var ms = new MemoryStream();
        await Request.Body.CopyToAsync(ms);
        ms.Position = 0;
        var binaryReader = new BinaryReader(ms);
        var modelLength = binaryReader.ReadInt32();
        var model = new int[modelLength];
        for (var i = 0; i < modelLength; i++)
        {
            model[i] = binaryReader.ReadInt32();
        }
        var inputKey = binaryReader.ReadInt64();
        
        var synapseLength = binaryReader.ReadInt32();
        var synapses = new float[synapseLength];
        for (var i = 0; i < synapseLength; i++)
        {
            synapses[i] = (float)binaryReader.ReadDouble();
        }
        var inputs = InputsCache[inputKey];
        var outputs = await _calcer.Calc(inputs.data, inputs.indexes, synapses, model);
        return new()
        {
            Outputs = outputs
        };
    }
}