import streamlit as st
import pandas as pd
import subprocess
import os

def run_and_display_stdout(*cmd_with_args, cwd):
    result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE, cwd=cwd)
    for line in iter(lambda: result.stdout.readline(), b""):
        str = line.decode('utf-8')
        if str.startswith("Epoch"):
          st.write(str.replace("\n", ""))

def main():
    st.set_page_config (layout="wide")
    scenariodir = os.path.dirname(os.path.realpath(__file__))

    os.environ['AZURE_RESOURCE_GROUP'] = "depa-ccr-test"
    os.environ['AZURE_STORAGE_ACCOUNT_NAME'] = "depacovidteststorage1"
    os.environ['AZURE_ICMR_CONTAINER_NAME'] = "icmrcontainer"
    os.environ['AZURE_COWIN_CONTAINER_NAME'] = "cowincontainer"
    os.environ['AZURE_INDEX_CONTAINER_NAME'] = "indexcontainer"
    os.environ['AZURE_MODEL_CONTAINER_NAME'] = "modelcontainer"
    os.environ['AZURE_OUTPUT_CONTAINER_NAME'] = "outputcontainer"
    os.environ['CONTAINER_REGISTRY'] = "sahilkathpal"
    os.environ['TOOLS_HOME'] = "/root/depa-training/external/confidential-sidecar-containers/tools"
    os.environ['CONTRACT_SERVICE_URL'] = "https://165.22.220.72:8000"
    os.environ['AZURE_KEYVAULT_ENDPOINT'] = "depa-covid-key-vault-1.vault.azure.net"

    st.title('DEPA for Training Demo')
    st.caption('Sample cowin data')
    cowin = pd.read_csv(scenariodir + "/data/cowin/poc_data_cowin_data.csv")
    st.dataframe(cowin)

    st.caption('Sample ICMR data')
    icmr = pd.read_csv(scenariodir + "/data/icmr/poc_data_icmr_data.csv")
    st.dataframe(icmr)

    st.caption('Sample state war room data')
    index = pd.read_csv(scenariodir + "/data/index/poc_data_statewarroom_data.csv")
    st.dataframe(index)

    st.divider()
    st.header('Publish Encrypted Datasets (TDP)')

    if st.button('De-identify datasets'):
        result = subprocess.run(["bash", "./preprocess.sh"], 
                                cwd=scenariodir + "/deployment/docker")
        if result.returncode == 0:
            st.success("De-identification successful")

        st.caption('De-identified cowin data')
        cowin = pd.read_csv(scenariodir + "/data/cowin/preprocessed/dp_cowin_standardised_anon.csv")
        st.dataframe(cowin)

        st.caption('Sample ICMR data')
        icmr = pd.read_csv(scenariodir + "/data/icmr/preprocessed/dp_icmr_standardised_anon.csv")
        st.dataframe(icmr)

        st.caption('Sample state war room data')
        index = pd.read_csv(scenariodir + "/data/index/preprocessed/dp_index_standardised_anon.csv")
        st.dataframe(index)
        

    if st.button('Generate and import keys'):
        result = subprocess.run(["bash", "3-import-keys.sh"],
                                cwd=scenariodir + "/data")
        if result.returncode == 0:
            st.success("Key generation and import successful")

    if st.button('Encrypt and upload datasets'):
        result = subprocess.run(["bash", "4-encrypt-data.sh"],
                                cwd=scenariodir + "/data")
        
        run_and_display_stdout("bash", "5-upload-encrypted-data.sh", cwd = scenariodir + "/data")

    st.divider()
    st.header('Sign and Register Contract (TDP and TDC)')

    tdp = st.text_input('TDP identity')
    tdc = st.text_input('TDC identity')
    tdp_vault = st.text_input('TDP Key Vault')

    if st.button('Show DID Documents'):
        os.environ['TDP_USERNAME'] = tdp
        os.environ['TDC_USERNAME'] = tdc
        os.environ['TDP_KEYVAULT'] = tdp_vault
        os.environ['CONTRACT_URL'] = os.environ['CONTRACT_SERVICE_URL']

        result = subprocess.run(["curl", "https://" + tdp + ".github.io/.well-known/did.json"], 
                                cwd="/root/depa-training/external/contract-ledger",
                                capture_output=True)
        st.write("TDP DID Document")
        st.json(result.stdout.decode('utf-8'))

        result = subprocess.run(["curl", "https://" + tdc + ".github.io/.well-known/did.json"], 
                                cwd="/root/depa-training/external/contract-ledger",
                                capture_output=True)
        st.write("TDC DID Document")
        st.json(result.stdout.decode('utf-8'))

    if st.button('Generate Contract'):
        result = subprocess.run(["bash", "demo/contract/update-contract.sh"], 
                                cwd="/root/contract-ledger")
        f = open("/root/contract-ledger/tmp/contracts/contract.json")
        st.json(f.read())

    if st.button('Sign and register contract as TDP'):
        result = subprocess.run(["bash", "demo/contract/3-sign-contract.sh"], 
                                cwd="/root/contract-ledger")
        result = subprocess.run(["bash", "demo/contract/4-register-contract.sh"], 
                                cwd="/root/contract-ledger", 
                                capture_output=True)
        if result.returncode == 0:
          st.write(result.stdout)

    contract = st.text_input('Contract ID')
    if st.button('Sign and register contract as TDC'):
        result = subprocess.run(["bash", "demo/contract/8-retrieve-contract.sh", contract], 
                                cwd="/root/contract-ledger")
        result = subprocess.run(["bash", "demo/contract/9-sign-contract.sh", contract], 
                                cwd="/root/contract-ledger")
        result = subprocess.run(["bash", "demo/contract/10-register-contract.sh"], 
                                cwd="/root/contract-ledger",
                                capture_output=True)
        if result.returncode == 0:
          st.write(result.stdout)

    st.divider()
    st.header('Train Model in CCR (TDC)')

    model = '''class SimpleModel(nn.Module):
                def __init__(self, input_dim):
                    super(SimpleModel, self).__init__()
                    self.fc1 = nn.Linear(input_dim, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, 1)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            # Step 5: Choose a loss function and optimizer
            model = SimpleModel(input_dim=train_features.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)'''

    st.caption('Sample model')
    st.subheader('Export model')
    st.code(model, language='python')

    if st.button('Export, encrypt and upload model'):
        result = subprocess.run(["bash", "save-model.sh"],
                                cwd=scenariodir + "/deployment/docker")

        if result.returncode == 0:
            st.success("Model exported in ONNX format")

    st.subheader('Train model in CCR')
    st.write('Sample model pipeline configuration')
    f = open(scenariodir + "/config/pipeline_config.json")
    st.json(f.read())

    contract = st.text_input('Signed contract ID')
    
    if st.button('Deploy CCR'):
        result = subprocess.run(["bash", "deploy.sh", "-c", contract, "-p", "../../config/pipeline_config.json"], 
                                cwd = scenariodir + "/deployment/aci")
        if result.returncode == 0:
            st.success("CCR deployed")

    if st.button('Get CCR logs'):
        run_and_display_stdout("bash", "az", "container", "logs", "--name", "depa-training-covid", "--resource-group", os.environ['AZURE_RESOURCE_GROUP'],  "--container-name", "depa-training", cwd = scenariodir + "/deployment/aci")

    if st.button('Export trained model'):
        result = subprocess.run(["bash", "6-download-decrypt-model.sh"], 
                                cwd=scenariodir + "/data")
        if result.returncode == 0:
            st.success("Model ready to download")


if __name__ == "__main__":
    main()
