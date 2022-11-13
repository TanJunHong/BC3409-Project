import {ArrowBack, ArrowForward} from "@mui/icons-material";
import {IconButton} from "@mui/material";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import {useLocation, useNavigate} from "react-router-dom";

export default function Header() {
    const navigate = useNavigate();
    const location = useLocation();

    return (
        <AppBar position="static" style={{background: "#282c34", textAlign: "center"}}>
            <Toolbar>
                <IconButton aria-label="back" color="inherit" disabled={location.pathname === "/"}
                            onClick={() => navigate(-1)}><ArrowBack/></IconButton>
                <Typography variant="h5" component="div" sx={{flexGrow: 1}}>
                    News Summariser
                </Typography>
                <IconButton aria-label="forward" color="inherit"
                            onClick={() => navigate(1)}><ArrowForward/></IconButton>
            </Toolbar>
        </AppBar>
    );
}
