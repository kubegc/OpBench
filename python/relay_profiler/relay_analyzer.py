import util
import model_importer.local_nns as local_nns
import tvm

mod, params, input_shape, output_shape,inputs = local_nns.get_network("resnet-18")
lib = util.compile_without_log(mod, tvm.target.Target("cuda"),params)
util.analyze_relay_json(lib.get_graph_json())