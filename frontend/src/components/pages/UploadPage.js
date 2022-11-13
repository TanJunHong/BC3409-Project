import {styled, TextField} from "@mui/material";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import React, {useState} from "react";
import {trackPromise, usePromiseTracker} from "react-promise-tracker";
import {useNavigate} from "react-router-dom";
import LoadingIndicator from "../LoadingIndicator";
import {Backend_URL} from "../Utils";
import {CategoriesPageLink} from "./CategoriesPage";

export const UploadPageLink = "/upload";

const CssTextField = styled(TextField)({
    "& .MuiOutlinedInput-root": {
        "& fieldset": {
            borderColor: "white"
        },
        "&:hover fieldset": {
            borderColor: "#1976d2"
        }

    },
    "& .MuiFormHelperText-root": {
        color: "white"
    }
});

export default function UploadPage() {
    const [selectedFile, setSelectedFile] = useState();
    const [isFilePicked, setIsFilePicked] = useState(false);
    const [datasetName, setDatasetName] = useState("");
    const navigate = useNavigate();
    const {promiseInProgress} = usePromiseTracker();


    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setIsFilePicked(true);
    };

    function handleSubmission() {
        const formData = new FormData();
        formData.append("File", selectedFile);
        formData.append("dataset", datasetName);
        trackPromise(
            fetch(Backend_URL + "upload", {
                method: "POST",
                body: formData
            }).then(response => response.json())
                .then(result => {
                    if (!("status" in result && result["status"] === "success")) {
                        //error!
                    }
                    navigate(CategoriesPageLink, {state: {dataset: datasetName}});
                }));
    }

    return (promiseInProgress ? <LoadingIndicator/> :
            <React.Fragment>
                <Button variant="contained" component="label">
                    Choose File
                    <input type="file" name="file" hidden onChange={handleFileChange} accept=".csv, application/JSON"/>
                </Button>
                {isFilePicked && selectedFile !== undefined ? selectedFile["name"].endsWith(".csv") || selectedFile["name"].endsWith(".json") ? (
                    <div>
                        <br/>
                        <Typography variant="h5">{selectedFile["name"]}</Typography>
                        <Typography>File Type: {selectedFile["type"]}</Typography>
                        <Typography>Size in bytes: {selectedFile["size"]}</Typography>
                        <Typography>
                            Last Modified Date: {selectedFile["lastModifiedDate"].toLocaleDateString()}
                        </Typography>
                        <br/>
                        <CssTextField id="outlined-helpertext" label="Dataset Name" variant="outlined"
                                      sx={{input: {color: "white"}}} InputLabelProps={{
                            style: {color: "#fff"}
                        }} value={datasetName} onChange={event => setDatasetName(event.target.value)}
                                      helperText="Overwrite any existing dataset with the same name"/>
                        <br/>
                        <br/>
                        <Button variant="contained" onClick={handleSubmission}>Submit</Button>
                    </div>
                ) : <div>
                    <br/>
                    <Typography>Please use only json or csv files!</Typography>
                </div> : ""}

            </React.Fragment>
    );
}
