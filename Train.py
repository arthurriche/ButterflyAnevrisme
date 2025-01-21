from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("tensorboard")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = '/content/drive/MyDrive/IDSC/4Students_AnXplore03/'
dataset = Dataset(folder_path)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2,
)

model = EncodeProcessDecode(
    node_input_size=6,
    edge_input_size=0,
    message_passing_num=15,
    hidden_size=128,
    output_size=3,
) #
loss = L2Loss() #
simulator = Simulator(
    node_input_size=6,
    edge_input_size=0,
    output_size=3,
    feature_index_start=0,
    feature_index_end=4,
    output_index_start=0,
    output_index_end=4,
    node_type_index=5,
    batch_size=1,
    model=model,
    device=device,
    model_dir="checkpoint/simulator.pth",
    time_index=4
) #
optimizer = torch.optim.Adam(simulator.parameters(), lr=0.0001)

train_epoch = TrainEpoch(
    model=simulator,
    loss=loss,
    optimizer=optimizer,
    device=device,
    verbose=True,
    starting_step=0,
    use_sub_graph=False,
)  #

for i in range(0, 10):
    print("\nEpoch: {}".format(i))
    print("=== Training ===")
    train_loss = train_epoch.run(train_loader, writer, "model.pth")

    writer.add_scalar("Loss/train/mean_value_per_epoch", train_loss, i)
    writer.flush()
    writer.close()
