import React from "react";
import {Route, Routes} from "react-router-dom";
import Header from "./components/Header";
import ArticlesPage, {ArticlesPageLink} from "./components/pages/ArticlesPage";
import CategoriesPage, {CategoriesPageLink} from "./components/pages/CategoriesPage";
import HomePage, {HomePageLink} from "./components/pages/HomePage";
import SummaryPage, {SummaryPageLink} from "./components/pages/SummaryPage";
import UploadPage, {UploadPageLink} from "./components/pages/UploadPage";

export default function App() {
    return (
        <React.Fragment>
            <Header/>
            <div style={{
                alignItems: "center",
                textAlign: "center",
                backgroundColor: "#282c34",
                color: "white",
                display: "flex",
                flexDirection: "column",
                fontSize: "calc(10px + 2vmin)",
                justifyContent: "center",
                minHeight: "100vh"
            }}>
                <Routes>
                    <Route path={HomePageLink} element={<HomePage/>}/>
                    <Route path={UploadPageLink} element={<UploadPage/>}/>
                    <Route path={CategoriesPageLink} element={<CategoriesPage/>}/>
                    <Route path={ArticlesPageLink} element={<ArticlesPage/>}/>
                    <Route path={SummaryPageLink} element={<SummaryPage/>}/>
                </Routes>
            </div>
        </React.Fragment>
    );
}
