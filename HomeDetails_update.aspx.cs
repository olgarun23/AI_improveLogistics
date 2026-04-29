using System;
using System.Data;
using System.Configuration;
using System.Collections;
using System.Web;
using System.Web.Security;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Web.UI.WebControls.WebParts;
using System.Web.UI.HtmlControls;
using System.Drawing;
using System.Collections.Generic;
using System.Text;
using SAMBAND_OR;
using System.Text.RegularExpressions;
using System.Linq;



public partial class _Default : System.Web.UI.Page
{
    public int m_nFnum = 0;
    public string m_strJsonList;

    protected static readonly log4net.ILog logger = log4net.LogManager.GetLogger(typeof(_Default));


    protected void Page_Load(object sender, EventArgs e)
    {
        m_divWarning.Visible = false;

        HtmlGenericControl body = (HtmlGenericControl)Master.FindControl("pageBody");
        body.Attributes.Add("onload", "loadUserInfo()");
        
		try
		{
	        if (!this.IsPostBack)
	        {
		        FillForm();
	        }

            initMacValidateData();
        }
        catch (Exception exe)
        {
            logger.Error("Error loading page ", exe);
            forwardToErrorPage(exe.Message);
        }
    }

	protected void FillForm()
	{
        m_divWarning.Visible = false;
		int iFNUM = Convert.ToInt32(Request.QueryString.Get("FNUM"));
		LoadDocument LD = new LoadDocument();
		DataRow dr = null;
        //List<TownData> townData = new List<TownData>();
        string ponReady = LD.getPonReady(iFNUM);
        List<Deilir> deilir = new List<Deilir>();
        List<PON> pon = new List<PON>();
        HouseData houseInfo = null;
        String cpeConn = null;
        String indoorDesc = null;
        m_lbtOpenInCrmByFnum.Attributes.Add("onclick", Utils.openInCrmByFnum(iFNUM));


        //husaskodunarmix
        WmHHServices.HHServicesWSD_PortTypeClient hh = new WmHHServices.HHServicesWSD_PortTypeClient();
        hh.ClientCredentials.UserName.UserName = ConfigConstants.WMUser;
        hh.ClientCredentials.UserName.Password = ConfigConstants.WMPass;
        hh.ClientCredentials.Windows.AllowedImpersonationLevel = System.Security.Principal.TokenImpersonationLevel.Impersonation;
         

        string afhBladLocation = "";
        string fnum = (Request.QueryString.Get("FNUM"));
        WmHHServices.result oRes = hh.getAfhBlodPDFLocation(fnum, out afhBladLocation);

        if(oRes.errorCode == "0")
        {
            string link = "<a target=\"_blank\" href=\"" + afhBladLocation + "\"><img src=\"./images/pdf32x32.png\" title=\"Afhendingarblað\"></a>";
            m_lblAfhBlodResult.Text = link;
            m_lblAfhBlodResult.Visible = true;
            m_lblAfhBlod.Visible = true;
        }
        else
        {
            m_lblAfhBlod.Visible = false;
            m_lblAfhBlodResult.Visible = false;
        }

        try
        {
            dr = LD.getDetails(SrchType.Homes, iFNUM);
            houseInfo = LD.getHouseByFnum(iFNUM);
            cpeConn = LD.getConnReport1(iFNUM);
            m_indoorDesc.Text = LD.getIndoorDescription(iFNUM); //þetta má vera empty

            //sölustaða undir staða húss
            int salesStatus = LD.getSalesStatusWithExplanation(iFNUM);
            if (salesStatus == -1)
            { m_salesStatusText.Text = "Villa kom upp";}
            else if (salesStatus == 1)
            { m_salesStatusText.Text = "1 - Auðkennt";}
            else if (salesStatus == 2)
            { m_salesStatusText.Text = "2 - Hafnað";}
            else if (salesStatus == 3)
            { m_salesStatusText.Text = "3 - Í framkvæmd"; }
            else if (salesStatus == 4)
            { m_salesStatusText.Text = "4 - Nær ljósleiðaratengdur";}
            else if (salesStatus == 5)
            { m_salesStatusText.Text = "5 - Ljósleiðaratengt en vv. ekki tengdur";}
            else if (salesStatus == 6)
            { m_salesStatusText.Text = "6 - Ljósleiðaratengt og vv. tengdur";}
            else
            { m_salesStatusText.Text = "Villa kom upp";}

            //heinum linkur sem var beðið um
            int heinum = LD.getHeinumByFnum(iFNUM);
            if (heinum > 0)
            {
                m_lbtnLuksja.ToolTip = "Opna í Luksjá";
                m_lbtnLuksja.Enabled = true;
                m_lbtnLuksja.Text = heinum.ToString();
            }
            else
            {
                m_lbtnLuksja.Enabled = false;
                m_lbtnLuksja.Text = "Heinum vantar!";
            }

            if (cpeConn == null)
            {
                m_lblConnReport.Visible = false;
                m_txtBoxConnReport.Visible = false;
            }
            else
            {
                //cpeConn = Regex.Replace(cpeConn, @"\t|\n|\r", "");
                m_lblConnReport.Visible = true;
                m_txtBoxConnReport.Visible = true;
                m_txtBoxConnReport.Text = cpeConn;
            }
        }
        catch (Exception e)
        {
            logger.Error("Error loading page ", e);
            forwardToErrorPage(e.Message);
            return;
        }

		HomeData home = new HomeData(dr);

        string userAllowedToChange = System.Configuration.ConfigurationManager.AppSettings["usernameAllowedToChangeRymi"];
        string username = Request.LogonUserIdentity.Name.Substring(Request.LogonUserIdentity.Name.IndexOf("\\")+1);

        userAllowedToChange = userAllowedToChange.ToLower();
        username = username.ToLower();

        if (userAllowedToChange.Split(new char[] { ';' }).Contains(username))
        {
            m_ddlHomeType.Enabled = true;
        }
        else
        {
            m_ddlHomeType.Enabled = false;
        }
        
        //m_cmbCpeTypes.DataSource = LD.getCpeTypes();
        m_cmbCpeTypes.DataSource = LD.getCpeTypes2();
        m_cmbCpeTypes.DataTextField = "cpeDesc";
        m_cmbCpeTypes.DataValueField = "cpetypeid";
        m_cmbCpeTypes.SelectedValue = home.CpeTypeId+"";
        m_cmbCpeTypes.DataBind();
        
        m_cmbSfpSwitch.DataSource = LD.getSfpSpeed();
        m_cmbSfpSwitch.DataTextField = "LABEL";
        m_cmbSfpSwitch.DataValueField = "ID";
        m_cmbSfpSwitch.DataBind();
        m_cmbSfpSwitch.SelectedValue = home.SfpSpeedSwitch.ToString();


        m_cmbIdregid.Checked = home.IdregidFullsplaest;


        if(home.FirstCustomerDate != DateTime.MinValue )
        {
            m_lblFirstCustomerDate.Text = home.FirstCustomerDate.ToString(StaticConstClass.DATE_DISPLAY_PATTERN);
            m_lblFirstCustomerlbl.Visible = true;
        }
        if(home.SalesStateDate != DateTime.MinValue)
        {
            m_lblSalesStateDate.Text = home.SalesStateDate.ToString(StaticConstClass.DATE_DISPLAY_PATTERN);
        }

        if (home.SplitConn != 0)
        {
            m_lblSplitConn.Text = "Já";
        }
        else
        {
            m_lblSplitConn.Text = "Nei";
        }


        m_lblDreifistodSpeed.Text = home.StationSpeed;

        m_lblIbudText.Text = home.Ibud; 
		Session.Add("IBUD", m_lblIbudText.Text);

		m_lblObjectID.Text = home.ObjectID.ToString();
		m_lblFNUM.Text = home.FNUM.ToString();

        m_nFnum = home.FNUM;
        if (home.Street != null)
        {
            m_lbtnAddress.Text = home.Street;
        }
        if (home.Town != null)
        {
            m_lblTown.Text = home.Town;
        }

		if (home.Tengt_NT == Convert.ToInt32(Acceptance.Accepted))
		{
			m_lblConnNTYN.Text = "Já";
			m_lblConnNTYN.ForeColor = Color.LimeGreen; 
		}
        else if (home.Tengt_NT == Convert.ToInt32(Acceptance.Denied))
		{
			m_lblConnNTYN.Text = "Nei";
			m_lblConnNTYN.ForeColor = Color.Red;
			m_chkConnHT.Enabled = false;
           // m_lblRor.Enabled = false;
            m_chkNetworkConnected.Enabled = false;
            m_chkConnectedInStation.Enabled = false;
            m_chkNetadgangst.Enabled = false;
            m_txtboxCPE_MAC.Enabled = false;
            m_chkConnHT.Enabled = false;
            m_chkNaerLjosl.Enabled = false;
           
            //m_txtDesc.Enabled = false;
            m_cmbCpeTypes.Enabled = false;


		}
		else
		{
			m_lblConnNTYN.Text = "Ekki vitað";
			m_lblConnNTYN.ForeColor = Color.Red;
			m_chkConnHT.Enabled = false;
            //m_lblRor.Enabled = false;
            m_chkNetworkConnected.Enabled = false;
            m_chkConnectedInStation.Enabled = false;
            m_chkNetadgangst.Enabled = false;
            m_txtboxCPE_MAC.Enabled = false;
            m_cmbCpeTypes.Enabled = false;

        
        }

        m_lnkOrders.NavigateUrl = "SetupOrders.aspx?fnum=" + home.FNUM;

        //Classic example of how not to code
        DataSet dsOffer = LD.getUsedOffer(iFNUM);
        if (dsOffer != null
            && dsOffer.Tables != null
            && dsOffer.Tables.Count > 0
            && dsOffer.Tables[0].Rows.Count > 0
            && dsOffer.Tables[0].Rows[0]["CONNECTIONDATE"].GetType() != typeof(DBNull))
        {
            DateTime dsConn;
            DateTime dsFirst;

            if (   DateTime.TryParse(dsOffer.Tables[0].Rows[0]["CONNECTIONDATE"].ToString(),out dsConn)
                && DateTime.TryParse(dsOffer.Tables[0].Rows[0]["FIRSTBILLDATE"].ToString(), out dsFirst))
            {
                m_lblUsedOfferL.Visible = true;
                m_lblUsedOffer.Text = String.Format("Tengidagur: {0}, reikningsd.: {1}",dsConn.ToShortDateString(), dsFirst.ToShortDateString());
            }
        }

        //end

		// Þetta er stillt í smiðnum, ef heimili er nettengt í hnútpunkti 
        // þ.e. ljósleiðaratengt er þetta true
        if (home.HTFlag == true)
        {
            m_chkConnHT.Checked = true;
        }
        else
        {
            m_chkConnHT.Checked = false;
        }
		  
        if ((home.Tengt_HT == Convert.ToInt32(Nettengdur.Vill_ekki_nettengjast_en_er_nettengdur_i_hnutpunkti)) ||
            (home.Tengt_HT == Convert.ToInt32(Nettengdur.Vill_ekki_nettengjast_og_er_ekki_nettengdur_i_hnutpunkti)))
        {
            m_chkConnHT.Enabled = false;
            m_txtboxCPE_MAC.Visible = false;
            m_lblCPE_MAC.Visible = false;
        }

        if (ponReady == "N")
        {
            if (home.CPE_MAC != null)
            {
                m_txtboxCPE_MAC.Text = home.CPE_MAC.ToString();
                m_lblCPE_MAC.Visible = true;
                m_txtboxCPE_MAC.Visible = true;
                m_cmbSfpSwitch.Visible = true;
            }
            // Ef CPE_MAC er ekki null(þ.e. með mac addressu) þá er búið að nettengja heimilið
            else if (home.Netadgangstaeki == (int)GeneralStatus.Yes)
            {
                m_lblCPE_MAC.Visible = true;
                m_txtboxCPE_MAC.Visible = true;
                m_cmbSfpSwitch.Visible = true;
            }
        }
        // MAC VS SERIAL PÆLINGAR
        else if (ponReady == "J")
        {
            //
            string ponSerial = LD.getSerialByFnum(iFNUM);
            if (ponSerial != null)
            {
                m_txtboxSerial.Text = ponSerial.ToString();
                lbl_serial.Visible = true;
                m_txtboxSerial.Visible = true;
                m_chkNetworkConnected.Checked = true;
                m_chkNetadgangst.Checked = true;
                nettengtEnableDisable();

            }
            else
            {

            }
        }



        

        //TODO
        if (home.NaerLjosleidarat == (int)BoolFlag.yes)
        {
            m_lblNotInSale.Visible = true;
            m_chkNotInSale.Visible = true;

            m_chkNotInSale.Checked = (home.NotInSale == (int)BoolFlag.yes);
        }
        else if (m_chkConnHT.Checked == true)
        {
            m_lblNotInSale.Visible = true;
            m_chkNotInSale.Visible = true;

            m_chkNotInSale.Checked = (home.NotInSale == (int)BoolFlag.yes);
        }
        else
        {
            m_lblNotInSale.Visible = false;
            m_chkNotInSale.Visible = false;
        }
        m_lblOfferName.Text = home.OfferName;
        m_txtDesc.Text = home.Description;
        displayConnectionInfo(home );

        generalCheckBox(home.Netadgangstaeki, ref m_chkNetadgangst);
        generalCheckBox(home.NetworkConnected, ref m_chkNetworkConnected);
        generalCheckBox(home.ConnectedInStation, ref m_chkConnectedInStation);
        generalCheckBox(home.NaerLjosleidarat, ref m_chkNaerLjosl);


        m_ddlHomeType.SelectedValue = home.HomeTypeId + "";


        if (houseInfo != null)
        {
            if (houseInfo.GreinilConfirmed == Convert.ToInt32(Acceptance.Accepted))
            {
                m_lblGreinilConfirmed.Text = "Já";
                m_lblGreinilConfirmed.ForeColor = Color.LimeGreen;
            }
            else if (houseInfo.GreinilConfirmed == Convert.ToInt32(Acceptance.Denied))
            {
                m_lblGreinilConfirmed.Text = "Nei";
                m_lblGreinilConfirmed.ForeColor = Color.Red;
            }
            else
            {
                m_lblGreinilConfirmed.Text = "Ekki vitað";
                m_lblGreinilConfirmed.ForeColor = Color.Red;
            }

            if (houseInfo.HouseType == "F")
            {
                m_lblTegundText.Text = "Já";
                m_lblTegundText.ForeColor = Color.LimeGreen;
                generalYesNoLable(houseInfo.Greinilagnir, ref m_lblGreinilagnir);
            }
            else if (houseInfo.HouseType == "E")
            {
                m_lblTegundText.Text = "Nei";
                m_lblTegundText.ForeColor = Color.Red;
            }
            else
            {
                m_lblTegundText.Text = "?";
            }

            m_lblZip.Text = houseInfo.Pnr;
            generalYesNoLable(houseInfo.Ror, ref m_lblRor);
        }
        else 
        {
            displayWarning("Hús finnst ekki." );
        }


        if (string.IsNullOrWhiteSpace(home.Street))
        {
            displayError("Rými finnst ekki í fasteignamatinu. Annað hvort þarf að laga töfluna lukor.faste eða fela þetta rými og laga skráningu í NE.");
        }

        if (ponReady == "J")
        {
            //new deilir info List<Deilir>
            deilir = LD.getDeilirInfoByFnum(iFNUM); //upplýsingar um tiltekna pon tengingu frá húsi út í port í tengistöð (leggir)
            pon = LD.getPonInfo(iFNUM); // upplýsingar um allar pon tengingar í þessu húsi
            double totalDistance = 0;
            int ponCount = pon.Count();
            int deilirCount = deilir.Count();
            if (deilir.Count > 0)
            {
                for (int i = 0; i < deilir.Count; i++)
                {
                    //innanhússstregur -- fic er port í inntaki
                    if (i == 0)
                    {
                        m_lblCRM.Visible = false;
                        m_lblCRMText.Visible = false;
                        m_lblInntakTxt.Text = deilir[i].m_til_stadur;            
                        m_lblInntakLiturText.Text = deilir[i].m_litur;
                        m_lblPortInntakTxt.Text = deilir[i].m_til_chassis + " - " + deilir[i].m_til_port;
                        //m_lblInntakHillaTxt.Text = deilir[i].m_til_equipment;
                        m_lblInntakPatchTxt.Text = TranslatePatchValue(deilir[i].m_patch.ToString());

                    }
                    //frá inntaki í tengistöð 1 
                    if (i == 1)
                    {
                        //m_lblInfo20.Text = "Tengistöð 1:";

                        m_txtConnInfo.Visible = false;
                        m_txtStation1.Text = "Tengistöð 1: \n";
                        m_txtStation1.Text += "Tengistöð: " + deilir[i].m_tengistod + " \n";
                        m_txtStation1.Text += "Tengiupplýsingar inn: " + deilir[i].m_til_chassis + " - " + deilir[i].m_til_port + " \n";
                        m_txtStation1.Text += "Patch staða: " + TranslatePatchValue(deilir[i].m_patch.ToString()) + " \n";
                        //m_txtStation1.Text += "Patch staða: " + deilir[i].m_patch + " \n";
                        m_txtStation1.Text += "Fjarlægð frá húsi: " + deilir[i].m_listi_lengd + " metrar" + " \n";
                        totalDistance = totalDistance + double.Parse(deilir[i].m_listi_lengd);

                    }

                    //endar hér
                    if (i == 2 && deilirCount == 3)
                    {
                        m_txtStation1.Text += "Tengiupplýsingar út: " + deilir[i].m_fra_chassis + " - " + deilir[i].m_fra_port + " \n";
                        m_txtStation1.Text += "Port í sviss: " + deilir[i].m_til_chassis + " - " + deilir[i].m_til_port + " \n";
                    }

                    //úr einni tengistöð í aðra tengistöð
                    if (i == 2 && deilirCount > 3)
                    {
                        m_txtStation1.Text += "Tengiupplýsingar út: " + deilir[i].m_fra_chassis + " - " + deilir[i].m_fra_port + " \n";
                        m_txtStation1.Text += "\n" + "Tengistöð 2:" + "\n";
                        m_txtStation1.Text += "Tengistöð: " + deilir[i].m_tengistod + " \n";
                        m_txtStation1.Text += "Tengiupplýsingar inn: " + deilir[i].m_til_chassis + " - " + deilir[i].m_til_port + " \n";
                        m_txtStation1.Text += "Patch staða: " + TranslatePatchValue(deilir[i].m_patch.ToString()) + " \n";
                        totalDistance = totalDistance + double.Parse(deilir[i].m_listi_lengd);
                        m_txtStation1.Text += "Fjarlægð frá húsi: " + totalDistance.ToString() + " metrar" + " \n";
                    }

                    //við höfum 1 tengistöð og skoðum aktífa portið í henni
                    if (i == 3)
                    {
                        m_txtStation1.Text += "Tengiupplýsingar út: " + deilir[i].m_fra_chassis + " - " + deilir[i].m_fra_port + " \n";
                        m_txtStation1.Text += "Port í sviss: " + deilir[i].m_til_chassis + " - " + deilir[i].m_til_port + " \n";
                    }
                }
            }
            else
            {
                m_lblInntak.Visible = false;
                m_lblInntakTxt.Visible = false;
                m_lblPortInntak.Visible = false;
                m_lblPortInntakTxt.Visible = false;
                m_lblInntakLitur.Visible = false;
                m_lblInntakLiturText.Visible = false;
                m_txtStation1.Visible = false;
                if (home.ForskradCRM != null)
                {
                    if (home.ForskradCRM == "1")
                    {
                        m_lblCRMText.Text = "Skráð í CRM";
                    }
                    else if (home.ForskradCRM == "2")
                    {
                        m_lblCRMText.Text = "Uppfært í CRM";
                    }
                }
                else if (home.ForskradCRM == null)
                {
                    m_lblCRMText.Text = "Óskráð í CRM";
                }
            }

            if (ponCount > 0)
            {
                m_lblAppCount.Visible = true;
                m_lblPonReady.Visible = true;
                m_lblPonPortCount.Visible = true;
                m_lblActiveCustCount.Visible = true;
                m_lblAppCountText.Text = pon[0].m_fjoldi;
                m_lblPonReadyText.Text = pon[1].m_fjoldi;
                m_lblPonPortCountText.Text = pon[2].m_fjoldi;
                m_lblActiveCustCountText.Text = pon[3].m_fjoldi;
            }
        }


		this.ViewState["HT"] = home.Tengt_HT.ToString();
		this.ViewState["NT"] = home.Tengt_NT.ToString();
		Session.Add("Home", home);

        nettengtEnableDisable();
        if (ponReady == "J")
        {
            loadCpeConn2(LD, "SF");
        }
        else if (ponReady == "N")
        {
            loadCpeConn(LD, home.CpeConnCode);
        }
        loadReturnedCpe(LD, iFNUM);
    }

    private void loadCpeConn(LoadDocument ld, string strSelectedValue)
    {
        m_cmbCpeConn.DataSource = ld.getCpeConnection();
        m_cmbCpeConn.DataTextField = "name";
        m_cmbCpeConn.DataValueField = "code";
        m_cmbCpeConn.DataBind();
        
        m_cmbCpeConn.SelectedValue = strSelectedValue;
        m_cmbCpeConn.Enabled = string.IsNullOrEmpty(strSelectedValue);
        m_txtboxCPE_MAC.Attributes.Add("onblur","validateMac()");
    }

    private void loadCpeConn2(LoadDocument ld, string strSelectedValue)
    {
        m_cmbCpeConn.DataSource = ld.getCpeConnection2();
        m_cmbCpeConn.DataTextField = "name";
        m_cmbCpeConn.DataValueField = "code";
        m_cmbCpeConn.DataBind();


        
        m_cmbCpeConn.SelectedValue = strSelectedValue;
        m_cmbCpeConn.Enabled = string.IsNullOrEmpty(strSelectedValue);
        //það vantar nýtt validate fall -> validateSerial .. 
        string read = "";
    }

    private void loadReturnedCpe(LoadDocument ld, int nFnum)
    {
        DataSet ds = ld.getReturnedEquipments(nFnum);
        if (ds != null
            && ds.Tables != null
            && ds.Tables.Count > 0
            && ds.Tables[0].Rows != null
            && ds.Tables[0].Rows.Count > 0)
        {
            m_lblCpeReturnTitle.Visible = true;
            for(int i=0;i<ds.Tables[0].Rows.Count;i++)
            {
                m_lblCpeReturnMsg.Text += ds.Tables[0].Rows[i]["CPE_MAC"].ToString() + 
                    " skilað þann " + 
                    Convert.ToDateTime(ds.Tables[0].Rows[i]["RETURN_DATE"]).ToString("dd.MM.yyyy") + 
                    ", kt: " + 
                    ds.Tables[0].Rows[i]["ssn"] +
                    " "+
                    ds.Tables[0].Rows[i]["customer_name"] + "<br/>";
            }
        }
    }


    private void initMacValidateData()
    {
        LoadDocument ld = new LoadDocument();
        DataSet dsConn = ld.getCpeConnectionRegex();

        StringBuilder sb = new StringBuilder();

        if (dsConn != null && dsConn.Tables != null && dsConn.Tables.Count > 0 && dsConn.Tables[0].Rows != null && dsConn.Tables[0].Rows.Count >0)
        {
            sb.Append("{\"regexList\":[");
            bool bFirst = true;
            foreach (DataRow dr in dsConn.Tables[0].Rows)
            {
                if (!bFirst)
                {
                    sb.Append(" , ");
                }
                sb.Append("{\"regex\": \""      + dr["regex"].ToString().Replace("\\","\\\\\\\\")        + "\",");
                sb.Append(" \"conn_code\":\""   + dr["conn_code"].ToString()    + "\",");
                sb.Append(" \"cpetypeid\":\""   + dr["cpetypeid"].ToString()    + "\"}");
                bFirst = false;
            }
            sb.Append("]}");

            m_strJsonList = sb.ToString();
        }
    
    }


    private void displayWarning( string strErrorMessage)
    {
        m_divWarning.Visible = true;
        m_divWarning.diplayWarning(strErrorMessage);
    }

    private void displayError(string strErrorMessage)
    {
        m_divWarning.Visible = true;
        m_divWarning.diplayError(strErrorMessage);
    }



    private void forwardToErrorPage(string strErrorMessage)
    {
        String strUrl = @"./Error.aspx?ErrorString="+strErrorMessage;
        Server.Execute(strUrl);
    }

    private void generalYesNoLable(int nGeneralStatus, ref Label lblYesNo)
    {
        if (nGeneralStatus == (int)GeneralStatus.Yes)
        {
            lblYesNo.Text = "Já";
            lblYesNo.ForeColor = Color.LimeGreen;
        }
        else
        {
            lblYesNo.Text = "Nei";
            lblYesNo.ForeColor = Color.Red;
        }
    }

    private void generalCheckBox(int nGeneralStatus, ref CheckBox cbCheckControl)
    {
        if (nGeneralStatus == (int)GeneralStatus.Yes)
        {
            cbCheckControl.Checked = true;
        }
        else
        {
            cbCheckControl.Checked = false;
        }
    }

    private void displayConnectionInfo(HomeData home)
    {
        string strConnectionInfoText = "Tengistöð: \t";

        if (home.Hnutur != 0 && home.HeitiDreifist != null)
        {
            strConnectionInfoText += home.HeitiDreifist.ToString() + " (" + home.Hnutur.ToString() + ")";
    
        }
        else if (home.Hnutur != 0)
        {
            strConnectionInfoText += home.Hnutur.ToString();
        }
        else if (home.HeitiDreifist != null)
        {
            strConnectionInfoText += home.HeitiDreifist.ToString();
        }
        else
        {
        }
        
        strConnectionInfoText += "\nTengigrind: \t";

        if (home.Tengigrind != null)
        {
            strConnectionInfoText += home.Tengigrind;
        }

        strConnectionInfoText += "\nSpjald: \t";

        if (home.Spjald != null)
        {
            strConnectionInfoText += home.Spjald.ToString();
        }

        strConnectionInfoText += "\nTengirauf: \t";


        if (home.Tengirauf != null)
        {
            strConnectionInfoText += home.Tengirauf;
        }


        strConnectionInfoText += "\nLitur í inntaksboxi, túba: ";

        if (home.LiturTupa != null)
        {
            strConnectionInfoText += home.LiturTupa;
        }

        
        strConnectionInfoText += "\nLitur í inntaksboxi, fiber: ";

        if (home.LiturFiber != null)
        {
            strConnectionInfoText += home.LiturFiber;
        }
        strConnectionInfoText += "\nDual:\t\t";

        if (home.Dual != null)
        {
            strConnectionInfoText += home.Dual;
        }

        strConnectionInfoText += "\nFjarlægð frá stöð:\t";

        if (home.LeggurLengd != 0)
        {
            strConnectionInfoText += home.LeggurLengd.ToString("###0", System.Globalization.CultureInfo.CurrentCulture.NumberFormat) + " metrar";
        }

        strConnectionInfoText += "\nÞjónustulagnir:\t";

        if (home.Thjonustulagnir != 0)
        {
            strConnectionInfoText += "Eru Samnýttar";
        }

        strConnectionInfoText += string.Format("\nInntaksstaður: \t{0}", home.Inntaksstadur);

        if(!string.IsNullOrWhiteSpace(home.ConnReport1))
        {
            strConnectionInfoText += string.Format("\nTengiskýrsla 1\n{0}", home.ConnReport1);
        }
        if (!string.IsNullOrWhiteSpace(home.ConnReport2))
        {
            strConnectionInfoText += string.Format("\nTengiskýrsla 2\n{0}", home.ConnReport2);
        }


        m_txtConnInfo.Text = strConnectionInfoText;


        if (home.DagsBreytt != null)
        {
            m_txtboxLastChange.Text = home.DagsBreytt.ToString();
        }
        if (home.Texti != null)
        {
            m_lblNotkun.Text = home.Texti;
        }
        if (home.Numer_VE != 0)
        {
            m_lbtnProjectNumber.Text = home.Numer_VE.ToString() + " " + home.HeitiVerkefnis;
        }
    }

    private HomeData pageToObject(HomeData hdSessionData)
    {
        if (m_chkNetadgangst.Checked == true)
        {
            string tempMac = m_txtboxCPE_MAC.Text;

            if (string.IsNullOrEmpty(tempMac))
            {
                if (string.IsNullOrEmpty(m_cmbCpeConn.SelectedValue))
                {
                    hdSessionData.CpeConnCode = null;
                }
                else if (m_cmbCpeConn.SelectedValue == "--")
                {
                    hdSessionData.CpeConnCode = null;
                }
                else
                {
                    hdSessionData.CpeConnCode = m_cmbCpeConn.SelectedValue;
                    m_cmbCpeConn.Enabled = false;
                }
                hdSessionData.CPE_MAC = null;
            }
            else if (Utils.validateMac(ref tempMac))
            {
                m_txtboxCPE_MAC.Text = tempMac;
                hdSessionData.CPE_MAC = tempMac;

                string strConn = getCpeConnByMac(tempMac);

                m_cmbCpeConn.SelectedValue = strConn;
                hdSessionData.CpeConnCode = strConn;
            }
            else
            {
                throw new ArgumentException("CPE_MAC er ekki lögleg. Einungis 12 tölu- og bókstafir (A-F) eru leyfilegir.");
            }

            hdSessionData.SfpSpeedSwitch = Convert.ToInt32(m_cmbSfpSwitch.SelectedValue);

        }
        else if (m_chkNetadgangst.Checked == false)
        {
            hdSessionData.CPE_MAC = "";
            m_txtboxCPE_MAC.Text = "";
            hdSessionData.CpeConnCode = null;
            m_cmbCpeConn.SelectedValue = "--";
        }

        
        hdSessionData.Tengt_HT = checkBoxToGeneralStatus(m_chkConnHT.Checked);
        hdSessionData.Netadgangstaeki = checkBoxToGeneralStatus(m_chkNetadgangst.Checked);
        hdSessionData.NetworkConnected = checkBoxToGeneralStatus(m_chkNetworkConnected.Checked);
        hdSessionData.ConnectedInStation = checkBoxToGeneralStatus(m_chkConnectedInStation.Checked);
        hdSessionData.NaerLjosleidarat = checkBoxToGeneralStatus(m_chkNaerLjosl.Checked);
        hdSessionData.HomeTypeId = int.Parse( m_ddlHomeType.SelectedValue );

        //TODO
        if (m_chkNotInSale.Checked && m_chkNaerLjosl.Checked)
        {
            hdSessionData.NotInSale = (int)BoolFlag.yes;
        }
        else if (m_chkNotInSale.Checked && m_chkConnHT.Checked == true)
        {
            hdSessionData.NotInSale = (int)BoolFlag.yes;
            m_chkNotInSale.Checked = true;
        }
        else
        {
            hdSessionData.NotInSale = (int)BoolFlag.no;
            m_chkNotInSale.Checked = false;
        }

        hdSessionData.Description = m_txtDesc.Text;
        hdSessionData.CpeTypeId = Convert.ToInt32(m_cmbCpeTypes.SelectedValue);

        hdSessionData.IdregidFullsplaest = m_cmbIdregid.Checked;
        



        return hdSessionData;
    }




	/// <summary>
	///		Vistum niður nýjar upplýsingar skv. því sem er á forminu.  Uppfærum líka Customer Status í CRM
	///		ef honum hefur verið breytt hér.
	/// </summary>
	protected void m_btnSubmit_Click(object sender, EventArgs e)
	{
        try
        {
            m_divWarning.Visible = false;
            HomeData hdSessionData = (HomeData)Session.Contents["Home"];

            if (hdSessionData == null)
            {
                logger.Error("Session empty");
                forwardToErrorPage("Session er útrunnið, leitaðu aftur af rýminu og gerðu breytingarnar upp á nýtt");
            }
            //yfirskrift á reitnum, er ekki að ná að sækja nýja commentið
            hdSessionData.IndoorDesc = m_indoorDesc.Text;
            hdSessionData = pageToObject( hdSessionData );

            LoadDocument ldLukorDB = new LoadDocument();

            if ( !string.IsNullOrEmpty(hdSessionData.CPE_MAC) && 
                  ldLukorDB.macExist(hdSessionData.FNUM, hdSessionData.CPE_MAC))
            {
                logger.Error("Mac is registered to other home");
                throw new ArgumentException("CPE_MAC er skráð á annað rými");
            }

            //reynum að uppfæra reitinn alltaf, höldunm svo utan um breytingar í changes töflu
            if ( !string.IsNullOrEmpty(hdSessionData.IndoorDesc))
            {
                ldLukorDB.updateIndoorDesc(hdSessionData.FNUM, hdSessionData.IndoorDesc);
            }

            if(!string.IsNullOrEmpty(m_txtboxSerial.Text))
            {
                ldLukorDB.updateSerial(hdSessionData.FNUM, m_txtboxSerial.Text);
            }
            
            // Save home data FIRST so CPE_MAC is in the DB before markCpe sets NETTENGT=1
            try
            {
                HomeDocument hd = new HomeDocument(Request.LogonUserIdentity.Name);
                hd.updateHome(hdSessionData);
            }
            catch (Exception exe)
            {
                logger.Error("Error in db update " + exe.Message);
                String strUrl = @"./Error.aspx?ErrorString=" + "HH database update failed: " + exe.Message;
                Server.Transfer(strUrl);
                return;
            }

            CRMConnector crmConnection = new CRMConnector();
            CRMCustomerData oCRMCustData = crmConnection.getOssByHouseholdId(hdSessionData.FNUM + "");
            
            if ( !string.IsNullOrEmpty( oCRMCustData.errorCode ) && oCRMCustData.errorCode.Equals("2") )
            {
                logger.Error("Home is not in CRM, fnum: "+hdSessionData.FNUM);

                displayWarning("Íbúð er ekki skráð í CRM, breytingar á stöðum eru því ekki uppfærðar í CRM. ");
            }
            else if (string.IsNullOrEmpty(oCRMCustData.errorCode) || !oCRMCustData.errorCode.Equals("0"))
            {
                logger.Error("Error in crm lookup "+oCRMCustData.errorCode +" "+oCRMCustData.errorMessage +", fnum: " + hdSessionData.FNUM);

                displayError("Villa í uppfléttingu í CRM, kóði: " +
                    oCRMCustData.errorCode + " villa: " + oCRMCustData.errorMessage);
                return;
            }
            else
            {
                logger.Debug("CRM else, fnum: " + hdSessionData.FNUM);

                if (!string.IsNullOrEmpty(oCRMCustData.HouseholdID) )
                {
                    logger.Debug("CRM fnum not null or empty, fnum: " + hdSessionData.FNUM);

                    int nConnectionStatus = Utils.getConnectionStatus(hdSessionData.NetworkConnected, hdSessionData.Tengt_NT, m_chkConnHT.Checked, m_chkNaerLjosl.Checked ? 1 : 0);

                    logger.Debug("Connection status: " + nConnectionStatus +", net connected "+hdSessionData.NetworkConnected+ ", nt "+hdSessionData.Tengt_NT + ", ht "+m_chkConnHT.Checked + ", naer ljos "+m_chkNaerLjosl.Checked);
                    
                    if (oCRMCustData == null || nConnectionStatus != oCRMCustData.FibreConnected)
                    {
                        try
                        {
                            crmConnection.UpdateHomeSalesStatus(hdSessionData.FNUM + "", nConnectionStatus);
                            logger.Info("CRM updated, fnum: " + hdSessionData.FNUM);
                        }
                        catch (Exception exe)
                        {
                            logger.Info("UpdateHomeSalesStatus return value " + exe.Message);
                            displayWarning(exe.Message);
                        }
                    }


                    WmCrmMarkCpe wmCrmCPE = new WmCrmMarkCpe();
                    Result res = wmCrmCPE.markCpe(hdSessionData.FNUM, m_chkNetadgangst.Checked);
                    if (res.ErrorCode != (int)ErrorCode.SUCCESS)
                    {
                        logger.Error("Error registering CPE in crm: " + res.ErrorMessage);

                        displayError("Villa við að skrá netaðgangstæki í CRM " + res.ErrorMessage);
                        return;
                    }
                    else
                    {
                        logger.Debug("Updating cpe done");
                    }
                }
            }
            
            m_lblInfo.Text = "Breytingar vistaðar";
        }
        catch (Exception exce)
        {
            logger.Error("Error in submit function " + exce.Message);
            String strUrl = @"./Error.aspx?ErrorString=" + exce.Message+" \n Stack trace: "+exce.StackTrace;
            Server.Transfer(strUrl);
        }
    }

    private string getCpeConnByMac(string strMac)
    {
        LoadDocument ld = new LoadDocument();
        DataSet dsConn = ld.getCpeConnectionRegex();
        for(int i=0;i<dsConn.Tables[0].Rows.Count;i++)
        {
            if (Regex.IsMatch(strMac, dsConn.Tables[0].Rows[i]["REGEX"].ToString()))
            {
                return dsConn.Tables[0].Rows[i]["CONN_CODE"].ToString();
            }
        }
        return "SF";
    }


    private int checkBoxToGeneralStatus(bool bChecked)
    {
        if (bChecked)
            return (int)GeneralStatus.Yes;
        else
            return (int)GeneralStatus.DontKnow;
    }    

	protected void m_chkNetadgangst_CheckedChanged(object sender, EventArgs e)
	{
		if (m_chkNetadgangst.Checked == true)
		{
			m_lblCPE_MAC.Visible = true;
			m_txtboxCPE_MAC.Visible = true;
            m_cmbSfpSwitch.Visible = true;
		}
		else if (m_chkNetadgangst.Checked == false)
		{
			m_lblCPE_MAC.Visible = false;
			m_txtboxCPE_MAC.Visible = false;
            m_cmbSfpSwitch.Visible = false;
		}

        nettengtEnableDisable();
	}

	protected void m_btnPrint_Click(object sender, EventArgs e)
	{
		Server.Transfer(@"./HomeDetailPrint.aspx");
	}

	protected void m_btnProject_Click(object sender, EventArgs e)
	{
		HomeData HD = (HomeData)this.Session["Home"];
		String navString = @"./ProjectDetails.aspx?ProjectID=";
		navString += HD.Numer_VE.ToString();
		Server.Transfer(navString);
	}

    protected void m_btnLuksja_Click(object sender, EventArgs e)
    {
        HomeData HD = (HomeData)this.Session["Home"];
        String navString = "https://luk.or.is/hoghkortagluggi/hh.html?heinum=";
        navString += m_lbtnLuksja.Text;
        //Server.Transfer(navString);
        Response.Redirect(navString);
    }


    protected void m_btnGetHouse_Click(object sender, EventArgs e)
	{
		HomeData HD = (HomeData)this.Session["Home"];
		String navString = @"./HouseDetails.aspx?";
        if (HD.HusId != 0)
        {
            navString += ("ID=" + HD.HusId);
        }
        else
        {
            navString += "ADDRESS=";
		    if (HD.TownCode == 0)
			    navString += "0000";
		    else
			    navString += HD.TownCode;

		    navString += " ";
		    navString += HD.Street;
        }
		Server.Transfer(navString);
	}

    protected void CPELabel_Click(object sender, EventArgs e)
    {
        ServiceBroker.HusHeimiliServices_webServicesService hhServicesForHH = new ServiceBroker.HusHeimiliServices_webServicesService();
        String strTemp = "";
        
        try
        {
            if (m_txtboxCPE_MAC.Text != null && m_txtboxCPE_MAC.Text.Trim() != "")
            {
                ServiceBroker.__result oRes = hhServicesForHH.pingCPEByMac(m_txtboxCPE_MAC.Text, out strTemp);
                m_txtPingResult.Visible = true;
                if (oRes.errorCode == ((int)ErrorCode.SUCCESS).ToString())
                {
                    if (strTemp == null || strTemp.Trim() == "")
                    {
                        m_txtPingResult.Text = "Ekki tókst að pinga tæki.";
                    }
                    else
                    {
                        m_txtPingResult.Text = strTemp;
                    }
                }
                else
                {
                    m_txtPingResult.Text = "Villa við að pinga CPE: " + oRes.errorMessage;
                }
            }
        }
        catch (Exception ex)
        {
            String strUrl = @"./Error.aspx?ErrorString=" + ex.Message;
            Server.Transfer(strUrl);
        }

    }
    protected void m_chkNaerLjosl_CheckedChanged(object sender, EventArgs e)
    {
        m_chkConnHT.Checked = false;
        nettengtEnableDisable();
    }
    protected void m_chkConnHT_CheckedChanged(object sender, EventArgs e)
    {
        m_chkNaerLjosl.Checked = false;
        nettengtEnableDisable();
    }
    protected void nettengtEnableDisable()
    {
        if (m_chkNetadgangst.Checked && m_chkConnHT.Checked && (m_lblConnNTYN.Text == "Nei" || m_lblConnNTYN.Text == "Ekki vitað" ))
        {
            m_chkNetworkConnected.Enabled = false;
        }
        else if (m_chkNetadgangst.Checked && m_chkConnHT.Checked )
        {
            m_chkNetworkConnected.Enabled = true;
        }
        else
        {
            m_chkNetworkConnected.Enabled = false;
            m_chkNetworkConnected.Checked = false;
        }
    }

    protected void m_chkConnectedInStation_CheckedChanged(object sender, EventArgs e)
    {
        nettengtEnableDisable();
    }

    protected void m_lbtChangeHistory_Click(object sender, EventArgs e)
    {
        LoadDocument ld = new LoadDocument();
        string[] strTables = { StaticConstClass.GV_TENGINGAR, StaticConstClass.GV_MOGULEGIR_VSKM };


        HomeData HD = (HomeData)this.Session["Home"];


        DataSet ds = ld.getChangeHistory(strTables, HD.FNUM);

        m_grdChangeHistory.DataSource = ds;
        m_grdChangeHistory.DataBind();
    }

    string TranslatePatchValue(string m_patch)
    {
        // Assuming m_patch is not null or empty, and contains single character values as per your requirement
        // It's a good practice to handle potential null or empty strings to avoid runtime errors
        if (string.IsNullOrEmpty(m_patch))
        {
            return "Óþekkt"; // Handle as needed for null or empty strings
        }

        m_patch = m_patch.ToUpper();

        switch (m_patch)
        {
            case "J":
                return "Tengt";
            case "X":
                return "Frátekið";
            case "N":
                return "Ótengt";
            case "A":
                return "Má aftengja";
            default:
                return "Óþekkt"; // Handle as needed for unexpected values
        }
    }


}
