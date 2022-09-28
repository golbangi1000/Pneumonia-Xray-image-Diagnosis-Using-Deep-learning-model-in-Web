package com.XrayWeb.project;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import javax.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.ModelAndView;

@Controller 
public class memberController {
	
	@Autowired 
	memberService MemberService;
	
	@Autowired
	xray_dataService XService;
	
	@RequestMapping(value="/signUp", method = RequestMethod.GET) 
	public ModelAndView create() {
	    return new ModelAndView("member/signUp");
	}
	
	@RequestMapping(value = "/signUp", method = RequestMethod.POST) 
	public ModelAndView createPost(@RequestParam Map<String, Object> map) {
	    ModelAndView mav = new ModelAndView();  

	    String member_no = this.MemberService.create(map); 
	   
	    if (member_no == null) {
	        mav.setViewName("redirect:member/signUp"); 
	    }else {
	        mav.setViewName("redirect:/login");
	    } 

	    return mav;
	}
	
	
	@RequestMapping(value="/login", method = RequestMethod.GET) 
	public ModelAndView login() throws IOException {
		
	    return new ModelAndView("member/login"); 
	}
	
	@RequestMapping(value = "/login", method = RequestMethod.POST) 
	public ModelAndView loginPost(@RequestParam Map<String, Object> map) {
			
		Map<String, Object> detailMap = this.MemberService.memberlogin(map); 
		ModelAndView mav = new ModelAndView(); 
		mav.addObject("data", detailMap); 
		try {
		String view_member_id = map.get("member_Id").toString(); 
		String db_member_Pwd = detailMap.get("member_Pwd").toString(); 
		
		System.out.println(view_member_id);
		System.out.println(db_member_Pwd); 

		List<Map<String, Object>> lists = XService.xray_dataList(map);
		
		System.out.println(lists);
		
		mav.addObject("lists", lists);
		
		
		mav.addObject("view_member_id", view_member_id); 
		mav.setViewName("/member/detail");
		
		} catch (NullPointerException e){
			mav.setViewName("/member/login");
		}
	    return mav;
	}
	
	
	@RequestMapping(value="/result", method = RequestMethod.GET)
	public ModelAndView result(@RequestParam Map<String, Object> map, HttpServletRequest request) {
		String member_id = request.getParameter("member_Id");
		String filename = request.getParameter("file_name");
		
		System.out.println(member_id);
		System.out.println(filename);
		
		//detail list gain
		ModelAndView mav = new ModelAndView(); 

		String path = "/resources/upload/file/"; 
		String user_file = path + member_id + "/" +filename.substring(0,filename.length()-4);
		
		mav.addObject("member_Id", request.getParameter("member_id"));

		mav.addObject("user_file", user_file);
		mav.setViewName("member/result");

		return mav;
	}
	
	
	@RequestMapping(value = "/result", method = RequestMethod.POST)
	public ModelAndView resultPost(@RequestParam Map<String, Object> map, HttpServletRequest request) {
		
		String member_id = request.getParameter("member_Id");
		String filename = request.getParameter("file_name");
		
		System.out.println(member_id);
		System.out.println(filename);
		
		ModelAndView mav = new ModelAndView();
	
	    return mav;
	}
}
