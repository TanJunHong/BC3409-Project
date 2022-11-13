import {FormControl, MenuItem, Select, styled} from "@mui/material";
import React, {useEffect, useState} from "react";
import {trackPromise, usePromiseTracker} from "react-promise-tracker";
import {useLocation, useNavigate} from "react-router-dom";
import ReactWordcloud from "react-wordcloud";
import "tippy.js/dist/tippy.css";
import LoadingIndicator from "../LoadingIndicator";
import {Backend_URL} from "../Utils";
import {ArticlesPageLink} from "./ArticlesPage";

export const CategoriesPageLink = "/categories";

const CssFormControl = styled(FormControl)({
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

export default function CategoriesPage() {
    // TODO: add slider to filter by date
    const navigate = useNavigate();
    const [words, setWords] = useState([]);
    const [dataset, setDataset] = useState("");
    const [datasets, setDatasets] = useState([]);
    const {state} = useLocation();
    const {promiseInProgress} = usePromiseTracker();

    function loadLabels(dataset) {
        const formData = new FormData();
        formData.append("dataset", dataset);
        trackPromise(
            fetch(Backend_URL + "load_labels", {
                method: "POST",
                body: formData
            }).then(response => response.json())
                .then(result => {
                    setWords(result["word_cloud"]);
                }));
    }

    function loadDatasets() {
        trackPromise(
            fetch(Backend_URL + "load_dataset_names", {
                method: "POST"
            }).then(response => response.json())
                .then(result => {
                    setDatasets(result["dataset_names"]);
                }));
    }

    function useMountEffect() {
        useEffect(() => {
            setDataset(state.dataset);
            loadLabels(state.dataset);
            loadDatasets();
        }, []);
    }

    const callbacks = {
        onWordClick: word => navigate(ArticlesPageLink, {
            state: {
                label: word["text"],
                dataset: dataset
            }
        })
    };

    const options = {
        colors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        fontFamily: "impact",
        fontStyle: "normal",
        fontWeight: "normal",
        fontSizes: [20, 80],
        enableTooltip: true
    };

    useMountEffect();

    return (promiseInProgress ? <LoadingIndicator/> :
            <React.Fragment>
                <CssFormControl fullWidth>
                    <Select
                        sx={{
                            color: "white"
                        }}
                        labelId="demo-simple-select-label"
                        id="demo-simple-select"
                        value={dataset}
                        onChange={event => {
                            setDataset(event.target.value);
                            loadLabels(event.target.value);
                        }}

                    >
                        {datasets.map((dataset, index) => (
                            <MenuItem key={index} value={dataset} sx={{
                                color: "black"
                            }}>
                                {dataset}
                            </MenuItem>
                        ))}
                    </Select>
                </CssFormControl>
                <ReactWordcloud callbacks={callbacks} options={options} words={words}/>
            </React.Fragment>
    );
}
