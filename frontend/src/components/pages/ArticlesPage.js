import {Grid, Typography} from "@mui/material";
import Button from "@mui/material/Button";
import React, {useEffect, useState} from "react";
import {trackPromise, usePromiseTracker} from "react-promise-tracker";
import {useLocation} from "react-router-dom";
import ArticleCard from "../ArticleCard";
import LoadingIndicator from "../LoadingIndicator";
import {Backend_URL} from "../Utils";

export const ArticlesPageLink = "/articles";

export default function ArticlesPage() {
    const [articles, setArticles] = useState([]);
    const [summary, setSummary] = useState([]);
    const {state} = useLocation();
    const label = state.label;
    const dataset = state.dataset;
    const limit = 6;
    const [skipAmount, setSkipAmount] = useState(0);
    const {promiseInProgress} = usePromiseTracker();

    function handleSkip(number) {
        setSkipAmount(number);
        loadArticles(number);
    }

    function capitalise(text) {
        return text.charAt(0).toUpperCase() + text.slice(1);
    }

    function loadArticles(skip) {
        const formData = new FormData();
        formData.append("dataset", dataset);
        formData.append("label", label);
        formData.append("skip", skip);
        formData.append("limit", limit);
        trackPromise(
            fetch(Backend_URL + "load_articles", {
                method: "POST",
                body: formData
            }).then(response => response.json())
                .then(result => {
                    setArticles(result["articles"]);
                    setSummary(result["summary"]);
                }));
    }

    function useMountEffect() {
        useEffect(() => {
            loadArticles(0);
        }, []);
    }

    useMountEffect();

    return (
        promiseInProgress ? <LoadingIndicator/> :
            <React.Fragment>
                <Grid container>
                    <Grid item xs={6}>
                        <Typography variant="h4">Summary</Typography>
                        <br/>
                        {summary.map((item, index) => (
                            <Typography paragraph={true} key={index} variant="body1"
                                        align="left">{capitalise(item)}</Typography>
                        ))}
                    </Grid>
                    <Grid item xs={6}>
                        <Typography variant="h4">Articles related to {label}</Typography>
                        <br/>
                        <Grid container justifyContent="center" alignItems="center" rowGap={2}>
                            {articles.map((article) => (
                                <ArticleCard key={article["_id"]} article={article}></ArticleCard>
                            ))}
                        </Grid>
                        <Button color="inherit"
                                onClick={() => handleSkip(Math.max(0, skipAmount - limit))}>Back</Button>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                        <Button color="inherit" onClick={() => handleSkip(skipAmount + limit)}>Forward</Button>
                    </Grid>

                </Grid>
            </React.Fragment>
    );
}
