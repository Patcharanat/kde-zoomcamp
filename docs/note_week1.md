# Week 1 - Document

## Docker

As data engineer, use for

- Reproducibility
- local experiment, local test
- Integration tests (CI/CD)

```python
import pandas as pd

# to get schema for a dataframe
pd.io.sql.get_schema(dataframe, name="table_name")
```

running docker single image (postgres) [using docker compose is more recommended]

```bash
docker run -it -e POSTGRES_USER="root" -e POSTGRES_PASSWORD="root" -e POSTGRES_DB="ny_taxi" -v ./src_zoomcamp:/var/lib/postgresql/data -p 5432:5432 postgres:13
```

running pgadmin

```bash
docker run -it -e PGADMIN_DEFAULT_EMAIL="admin@admin.com" -e PGADMIN_DEFAULT_PASSWORD="root" -p 8080:80 dpage/pgadmin4
```

```docker
# docker-compose.yml
services:
	pgadmin:
    image: dpage/pgadmin4:latest
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@admin.com
      - PGADMIN_DEFAULT_PASSWORD=root
    ports:
      - "8080:80"
```

transforming jupyter notebook to python code

```bash
jupyter nbconvert --to=script <filename.ipynb>
```

pgadmin UI setting

- connect via localhost:port
- server → registry → …
- Tools → query tool

*bug: installed postgresql in desktop → uninstall and rerun connect sqlalchemy*

## Google Cloud Platform (GCP) & Terraform

Install Terraform (add to global path in system variable)

install gcloud cli (sdk)

### Google Cloud (gcloud CLI, IAM&Admin, service account)

1. setting up service account in selected project 
    - (console → IAM&Admin → service account → create service account → manage key → json)
    
    1.5: setting environment for window
    
    - Use GitBash/MINGW to be able to use linux command line on window
2. get key (.json) to a wanted path (download)

```bash
export GOOGLE_APPLICATION_CREDENTIALS="<path/to/your/service-account-authkeys>.json"
#export GOOGLE_APPLICATION_CREDENTIALS="C:\Users\HP\Desktop\ken\Self-Learning\zoomcamp\kde_zoomcamp.json"

# Refresh token/session, and verify authentication
gcloud auth application-default login
```

1. set access for service account → IAM & admin
    - Click the *Edit principal* icon for your service account.
    - Add these roles in addition to *Viewer* : **Storage Admin** + **Storage Object Admin** + **BigQuery Admin**
2. Enable API
    - https://console.cloud.google.com/apis/library/iam.googleapis.com
    - https://console.cloud.google.com/apis/library/iamcredentials.googleapis.com

### Terraform (Configuration)

    Infrastructure manager (provisioning infrastructure resources)

like, create cloud resource for the project (e.g. creating bucket) using config .tf file

File for configuration

- .terraform-version
- main.tf
- variable.tf

General execution step:

1. `terraform init` :
    - Initializes & configures the backend, installs plugins/providers, & checks out an existing configuration from a version control
2. `terraform plan` :
    - Matches/previews local changes against a remote state, and proposes an Execution Plan.
3. `terraform apply` : Apply change to cloud
    - Asks for approval to the proposed plan, and applies changes to cloud
4. `terraform destroy` :
    - Removes your stack from the Cloud

Execution:

```bash
# Refresh service-account's auth-token for this session
gcloud auth application-default login

# Initialize state file (.tfstate)
terraform init

# Check changes to new infra plan
terraform plan -var="project=<your-gcp-project-id>"
```

```bash
# Create new infra
terraform apply -var="project=<your-gcp-project-id>"

# Delete infra after your work, to avoid costs on any running services
terraform destroy
```

## Set up Environments in Google Cloud

Be careful of billing → only run instance when we use, else just stop it

1. GCP → Compute Engine → Metadata
2. Generate SSH key → Add in Metadata tabs → SSH key
    - generating ssh key
        
        ```bash
        # git bash
        cd ~
        mkdir .ssh
        cd .ssh/
        
        # from official google cloud create ssh key
        ssh-keygen -t rsa -f <key_file_name> -C <username> -b 2048
        # ssh-keygen -t rsa -f gcp -C kde-user -b 2048
        
        # enter needed password to enter machine every time
        cat <key_file_name>.pub
        ```
        
    - copy all output to `Add SSH key` in VM -> Metadata
3. go back to VM instances → Create Instance
4. config everything include Boot disk (Ubuntu) → create instance
5. wait until finish → copy external Ip → command line
    
    ```bash
    ssh -i ~/.ssh/gcp the_name_used_when_generated_key@external_IP
    # ssh -i ~/.ssh/gcp kde-user@34.142.198.62
    ```
    
    and then move into VM’s bash
    
6. download anaconda for VM instance (Linux 64) and install into VM
    
    ```bash
    wget <download_link>
    
    bash <Anaconda_file_name>.sh
    ```
    
7. config server by going back to local git bash (or open new window) then:
    
    ```bash
    cd .shh/
    
    touch config
    
    code config
    ```
    
    ```
    # type in config file
    
    Host de-zoomcamp
    	HostName <external_IP>
    	User username_used_for_generated_key
    	IdentityFile c:/Users/alexe/.ssh/gcp
    ```
    
    and then run in GitBash to login into VM bash using ssh config file
    
    ```bash
    ssh <Host> # replace Host with de-zoomcamp we defined in config file
    ```
    
    Ctrl+D to log out
    
    ```bash
    less .bashrc
    # if there (base) appeared, it means anaconda works
    which python
    python
    ```
    
    finally we are able to use python in command line of VM’s bash
    
8. Install Docker
    
    ```bash
    sudo apt-get update
    sudo apt-get install docker.io
    ```
    
    if facing permission denied issue:
    
    ```bash
    sudo groupadd docker
    
    # Add the connected user $USER to the docker group
    # Optionally change the username to match your preferred user
    sudo gpasswd -a $USER docker
    
    sudo service docker restart
    
    # logout by Ctrl+D first and then,
    ssh <Host>
    ```
    
9. Install Docker Compose
    - go to docker compose github → latest → docker-compose-linux-x86_64 (copy link)
        
        ```bash
        # In VM bash
        # build new folder (bin)
        mkdir bin
        cd bin/
        
        # docker-compose: docker-compose-linux-x86_64
        wget <docker-compose-url> -O docker-compose
        
        # docker-compose should be display, but system don't know it's executable
        chmod +x docker-compose
        
        ./docker-compose version
        ```
        
        ```bash
        # In VM bash
        # build new folder (bin)
        mkdir bin
        cd bin/
        
        # docker-compose: docker-compose-linux-x86_64
        wget <docker-compose-url> -O docker-compose
        
        # docker-compose should be display, but system don't know it's executable
        chmod +x docker-compose
        
        ./docker-compose version
        ```
        
    - add docker-compose to path
        
        ```bash
        # back to main directory
        cd
        
        nano .bashrc
        
        export PATH="${HOME}/bin:${PATH}"
        ```
        
        then press Ctrl+O to save then Ctrl+X to exit
        
    - check if docker-compose is installed
        
        ```bash
        source .bashrc # to related login or log out?
        
        which docker-compose
        
        docker-compose version
        ```
        
10. get docker-compose file into VM
    
    in Zoomcamp tutorial → clone de-zoomcamp repo
    
    ```bash
    git clone <zoomcamp-url-repo>
    ```
    
    then cd to week1/2_docker_sql/ then
    
    ```bash
    docker-compose up -d
    
    # check docker running
    docker ps
    ```
    
11. configure VS code to use usual editor on remote machine
    - vscode → extension → remote ssh → install extension
    - remote window (Ctrl+Shift+P) → Remote SSH: connect to host → select host we defined in config file
12. still, working in VM’s bash
    
    ```bash
    conda install -c conda-forge pgcli
    
    pip install -U mycli
    
    pgcli -h localhost -U root -d ny_taxi
    # enter invisible password defined in `docker-compose.yml`
    
    \dt
    ```
    
    - if encounter `solving environment` loop, do this:
        
        ```bash
        # Ctrl+C to stop bash
        conda config --remove channels conda-forge
        conda config --add channels conda-forge
        conda install -c conda-forge pgcli
        
        # if not work try next, this should take a while
        conda update conda
        conda install -c conda-forge pgcli
        
        # if not work, try this: https://github.com/conda/conda/issues/9367#issuecomment-558863143
        # but the second worked for me
        ```
        
13. use vs code as development environment → select folder
    - Ctrl+~: open teminal in vscode
    - select PORTS section → Forward a Port → Port 5432
    - now we can use
        
        ```bash
        pgcli -h localhost -U root -d ny_taxi
        ```
        
        in local machine to interact with postgres (not via ssh)
        
        or even check pgadmin after add port pgadmin (8080)
        
    - then open upload_notebook.ipynb notebook to reproduce process in previous done in docker
        - use VM’s terminal to wget <url-data>
    - after connecting via notebook, and connect + upload schema and data, we can check by going back to VM’s terminal
        
        ```bash
        ssh <Host>
        
        pgcli -h localhost -U root -d ny_taxi
        
        \dt
        ```
        
        to check if data is loaded to postgres database in VM
        
    1. install Terraform in VM (Linux)
        
        ```bash
        cd bin/
        
        wget <download-url>
        
        sudo apt-get install unzip
        
        unzip <downloaded-file-name>.zip
        
        rm <downloaded-file-name>.zip
        
        ls # check terraform
        
        terraform -version
        ```
        
    2. use terraform in VM
        - cd go to home then go to de-zoomcamp dir → week1 → terraform_gcp → cd terraform/
        - transferring service account credential (which located in ./gc in local) using sftp
            - go back to <git bash>
                
                ```bash
                cd .gc/
                
                ls # suppose we have credentials.json here
                
                sftp <Host>
                
                mkdir .gc
                cd .gc
                
                put credentials.json
                ```
                
            - and then go back to VM’s bash
                
                ```bash
                export GOOGLE_APPLICATION_CREDENTIALS=~/.gc/ny-rides.json
                # export GOOGLE_APPLICATION_CREDENTIALS=<service_key_file_name>.json
                
                # authenticate
                gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
                # or
                gcloud auth application-default login
                ```
                
        - change `variable` variable in [variables.tf](http://variables.tf) and then
            
            ```bash
            terraform init
            
            terraform plan
            
            terraform apply
            
            yes
            ```
            
    3. don’t forget to stop stop VM instance or delete
        - but after reconnect external IP will change
            
            ```bash
            nano .ssh/config
            # or
            code .ssh/config
            ```
            
            then change external IP to the current show 
            
    
    Note: If can’t test on GCP with free credit → try github codespace

---