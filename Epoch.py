from tqdm import tqdm as tqdm
import sys
      
class Epoch:
    def __init__(
        self,
        model,
        loss,
        stage_name,
        parameters,
        device="cpu",
        verbose=True,
        starting_step=0,
    ):
        self.model = model
        self.loss = loss
        self.verbose = verbose
        self.device = device
        self.parameters = parameters
        self.step = 0
        self._to_device()

        self.full_batch_graph = []
        self.starting_step = starting_step

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, writer=None, model_save_dir="checkpoint/simulator.pth"):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for graph_data in iterator:
                for indx in range(1):
                    #TODO: check if we need this processing or not because it may already be the case
                    # input_graph = Data(
                    #     x=graph_data["x"][indx],
                    #     pos=graph_data["pos"][indx],
                    #     edge_index=graph_data["edge_index"][indx],
                    #     edge_attr=graph_data.get("edge_attr", [None])[indx],
                    #     y=graph_data["y"][indx],
                    # ).to(self.device)
                    input_graph = graph_data.to(self.device)

                    self.full_batch_graph.append(input_graph)

                if len(self.full_batch_graph) % self.model.batch_size == 0:

                    loss = self.batch_update(self.full_batch_graph, writer)

                    # update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {self.loss.__name__: loss_meter.mean}
                    logs.update(loss_logs)

                    if self.model.training:
                        writer.add_scalar(
                            "Loss/train/value_per_step",
                            loss_value,
                            self.step + self.starting_step,
                        )

                    else:
                        writer.add_scalar(
                            "Loss/test/value_per_step",
                            loss_value,
                            self.step + self.starting_step,
                        )

                    if self.step % 200 == 0:
                        self.model.save_checkpoint(model_save_dir)
                        writer.flush()

                    self.step += 1
                    self.full_batch_graph = []

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)

        return loss_meter.mean


class TrainEpoch(Epoch):
    def __init__(
        self,
        model,
        loss,
        parameters,
        optimizer,
        device="cpu",
        verbose=True,
        starting_step=0,
        use_sub_graph=False,
    ):
        super().__init__(
            model=model,
            loss=loss,
            stage_name="train",
            parameters=parameters,
            device=device,
            verbose=verbose,
            starting_step=starting_step,
        )
        self.optimizer = optimizer
        self.use_sub_graph = use_sub_graph

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, batch_graph, writer):
        self.optimizer.zero_grad()
        loss = 0
        #TODO: check that batch_graph is either a list of graphs of a list of one list of graphs
        for graph in batch_graph:
            
            node_type = graph['x'][:, self.model.node_type_index]
            network_output, target_delta_normalized = self.model(graph)
            loss += self.loss(
                target_delta_normalized,
                network_output,
                node_type,
            )

        loss /= len(batch_graph)
        loss.backward()
        max_norm = 10.0
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()

        return loss

  
